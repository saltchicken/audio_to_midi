use biquad::{Biquad, Coefficients, DirectForm2Transposed, ToHertz, Type, Q_BUTTERWORTH_F32};
use pitch_detection::detector::yin::YINDetector;
use pitch_detection::detector::PitchDetector;
use midly::{Header, Format, Timing, Track, TrackEvent, TrackEventKind, MidiMessage, Smf, MetaMessage, num::u7};
use std::env;
use std::path::Path;
use std::fs::File;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

fn hz_to_midi(hz: f64) -> f64 {
    69.0 + 12.0 * (hz / 440.0).log2()
}

fn frame_to_tick(frame: usize, sample_rate: u32) -> u32 {
    // Assuming 120 BPM and 480 Ticks Per Quarter Note (TPQN)
    // 1 beat = 0.5s = 480 ticks -> 960 ticks per second
    ((frame as f64 / sample_rate as f64) * 960.0).round() as u32
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <input_audio_file>", args[0]);
        std::process::exit(1);
    }
    
    let input_path = &args[1];
    let output_path = Path::new(input_path).with_extension("mid");

    println!("Loading audio from {}...", input_path);
    
    // 1. Open the media source.
    let file = Box::new(File::open(input_path).expect("Failed to open audio file"));
    let mss = MediaSourceStream::new(file, Default::default());

    // 2. Setup format hints based on file extension
    let mut hint = Hint::new();
    if let Some(ext) = Path::new(input_path).extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }

    // 3. Probe the media source to determine the format.
    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &fmt_opts, &meta_opts)
        .expect("Unsupported or unrecognized audio format");

    let mut format = probed.format;

    // 4. Find the first valid audio track.
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .expect("No supported audio tracks found");

    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.expect("Unknown sample rate");

    // 5. Create a decoder for the track.
    let dec_opts: DecoderOptions = Default::default();
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &dec_opts)
        .expect("Unsupported audio codec");

    let mut samples: Vec<f32> = Vec::new();
    let mut sample_buf = None;

    println!("Decoding audio and downmixing to mono...");
    // 6. Decode all packets and downmix channels
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(Error::IoError(_)) => break, // Expected End-of-file
            Err(e) => {
                eprintln!("Error reading packet: {}", e);
                break;
            }
        };

        if packet.track_id() != track_id {
            continue; // Skip packets belonging to other tracks
        }

        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                // Get the channel count directly from the decoded buffer
                let channel_count = audio_buf.spec().channels.count();

                // Initialize the sample buffer if it hasn't been set up yet
                if sample_buf.is_none() {
                    let spec = *audio_buf.spec();
                    let duration = audio_buf.capacity() as u64;
                    sample_buf = Some(SampleBuffer::<f32>::new(duration, spec));
                }

                if let Some(buf) = &mut sample_buf {
                    buf.copy_interleaved_ref(audio_buf);
                    
                    let interleaved = buf.samples();
                    
                    // Downmix to mono by averaging all channels for each frame
                    for frame in interleaved.chunks_exact(channel_count) {
                        let sum: f32 = frame.iter().sum();
                        samples.push(sum / channel_count as f32);
                    }
                }
            }
            Err(Error::DecodeError(e)) => {
                eprintln!("Decode error: {}", e);
            }
            Err(e) => {
                eprintln!("Fatal decode error: {}", e);
                break;
            }
        }
    }

    println!("Applying Bandpass Filter (150Hz - 1000Hz)...");
    let fs = sample_rate.hz();
    let f0_hp = 150.hz();
    let f0_lp = 1000.hz();
    
    let hp_coeffs = Coefficients::<f32>::from_params(Type::HighPass, fs, f0_hp, Q_BUTTERWORTH_F32).unwrap();
    let lp_coeffs = Coefficients::<f32>::from_params(Type::LowPass, fs, f0_lp, Q_BUTTERWORTH_F32).unwrap();
    
    let mut hp_filter = DirectForm2Transposed::<f32>::new(hp_coeffs);
    let mut lp_filter = DirectForm2Transposed::<f32>::new(lp_coeffs);

    // Apply filters and cast to f64 for YIN algorithm
    let filtered_samples: Vec<f64> = samples.into_iter()
        .map(|s| lp_filter.run(hp_filter.run(s)) as f64)
        .collect();

    println!("Extracting Pitch using YIN...");
    let window_size = 2048;
    let hop_size = 512; 
    let padding = window_size / 2;
    let mut detector = YINDetector::new(window_size, padding);
    let mut raw_pitches = Vec::new();

    for i in (0..filtered_samples.len().saturating_sub(window_size)).step_by(hop_size) {
        let window = &filtered_samples[i..i + window_size];
        
        // 1. Calculate RMS (Volume) to gate out background noise and reverb tails
        let mut rms = 0.0;
        for &s in window { rms += s * s; }
        rms = (rms / window.len() as f64).sqrt();

        // 2. Only run pitch detection if the audio is loud enough
        let volume_threshold = 0.015; // Roughly -36dB. Adjust if quiet notes get cut off.
        if rms > volume_threshold {
            if let Some(pitch) = detector.get_pitch(&window, sample_rate as usize, 0.15, 0.20) {
                raw_pitches.push((i, hz_to_midi(pitch.frequency), rms));
            } else {
                raw_pitches.push((i, 0.0, rms));
            }
        } else {
            raw_pitches.push((i, 0.0, rms)); // Silence
        }
    }

    println!("Applying Median Filter to smooth vibrato...");
    let median_window = 7; // Smooth over ~7 frames
    let mut smoothed_pitches = Vec::new();
    
    for i in 0..raw_pitches.len() {
        let start = i.saturating_sub(median_window / 2);
        let end = (i + median_window / 2).min(raw_pitches.len() - 1);
        
        // Median smooth pitch
        let mut window_p: Vec<f64> = raw_pitches[start..=end].iter().map(|&(_, p, _)| p).collect();
        window_p.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_pitch = window_p[window_p.len() / 2];

        // Median smooth RMS
        let mut window_rms: Vec<f64> = raw_pitches[start..=end].iter().map(|&(_, _, r)| r).collect();
        window_rms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_rms = window_rms[window_rms.len() / 2];

        smoothed_pitches.push((raw_pitches[i].0, median_pitch, median_rms));
    }

    println!("Quantizing with Hysteresis (Debouncing boundaries)...");
    let mut quantized_pitches = Vec::new();
    let mut current_quantized = 0u8;

    for &(frame_idx, midi_float, rms) in &smoothed_pitches {
        let q = if midi_float > 0.0 {
            // Hysteresis: Require a >0.6 semitone change to switch notes.
            // This stops vibrato from flipping back and forth between C and C#.
            if current_quantized > 0 && (midi_float - current_quantized as f64).abs() < 0.6 {
                current_quantized
            } else {
                midi_float.round() as u8
            }
        } else {
            0
        };
        current_quantized = q;
        quantized_pitches.push((frame_idx, q, rms));
    }

    println!("Segmenting Notes...");
    let min_stable_frames = 5; 
    let mut midi_events = Vec::new();
    
    // Tracks: (pitch, start_frame, peak_volume)
    let mut current_note: Option<(u8, usize, f64)> = None; 
    let mut candidate_note = 0u8;
    let mut candidate_count = 0;

    for &(frame_idx, pitch, rms) in &quantized_pitches {
        if pitch == candidate_note {
            candidate_count += 1;
        } else {
            candidate_note = pitch;
            candidate_count = 1;
        }

        if candidate_count == min_stable_frames {
            let actual_start_frame = frame_idx.saturating_sub(min_stable_frames * hop_size);

            match current_note {
                Some((curr_pitch, start_frame, curr_rms)) => {
                    if curr_pitch != candidate_note {
                        if curr_pitch != 0 {
                            midi_events.push((curr_pitch, start_frame, actual_start_frame, curr_rms));
                        }
                        if candidate_note != 0 {
                            current_note = Some((candidate_note, actual_start_frame, rms));
                        } else {
                            current_note = None;
                        }
                    } else {
                        // Maintain the peak volume for the currently held note
                        current_note = Some((curr_pitch, start_frame, curr_rms.max(rms)));
                    }
                }
                None => {
                    if candidate_note != 0 {
                        current_note = Some((candidate_note, actual_start_frame, rms));
                    }
                }
            }
        } else if let Some((curr_pitch, start_frame, curr_rms)) = current_note {
            // Track peak volume while holding the note
            if curr_pitch == pitch {
                current_note = Some((curr_pitch, start_frame, curr_rms.max(rms)));
            }
        }
    }

    if let Some((curr_pitch, start_frame, curr_rms)) = current_note {
        midi_events.push((curr_pitch, start_frame, filtered_samples.len(), curr_rms));
    }

    println!("Writing MIDI file...");
    let header = Header::new(Format::SingleTrack, Timing::Metrical(480.into()));
    let mut track = Track::new();
    let mut last_tick = 0;

    for (pitch, start_frame, end_frame, max_rms) in midi_events {
        let start_tick = frame_to_tick(start_frame, sample_rate);
        let end_tick = frame_to_tick(end_frame, sample_rate);

        // Map max_rms to MIDI velocity (0 - 127) to give the performance dynamics
        let vel_f = (max_rms.sqrt() * 300.0).clamp(30.0, 127.0);
        let velocity = vel_f as u8;

        let delta_on = start_tick.saturating_sub(last_tick);
        track.push(TrackEvent {
            delta: delta_on.into(),
            kind: TrackEventKind::Midi {
                channel: 0.into(),
                message: MidiMessage::NoteOn { 
                    key: u7::from(pitch), 
                    vel: u7::from(velocity) 
                }
            }
        });

        let delta_off = end_tick.saturating_sub(start_tick);
        track.push(TrackEvent {
            delta: delta_off.into(),
            kind: TrackEventKind::Midi {
                channel: 0.into(),
                message: MidiMessage::NoteOff { 
                    key: u7::from(pitch), 
                    vel: u7::from(0) 
                }
            }
        });

        last_tick = end_tick;
    }

    track.push(TrackEvent {
        delta: 0.into(),
        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack)
    });

    let mut smf = Smf::new(header);
    smf.tracks.push(track);
    smf.save(&output_path).unwrap();
    println!("Done! Saved to {}", output_path.display());
}
