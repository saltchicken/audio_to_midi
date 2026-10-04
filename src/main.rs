use hound;
use biquad::{Biquad, Coefficients, DirectForm2Transposed, ToHertz, Type, Q_BUTTERWORTH_F32};
use pitch_detection::detector::yin::YINDetector;
use pitch_detection::detector::PitchDetector;
use midly::{Header, Format, Timing, Track, TrackEvent, TrackEventKind, MidiMessage, Smf, MetaMessage, num::u7};

fn hz_to_midi(hz: f64) -> f64 {
    69.0 + 12.0 * (hz / 440.0).log2()
}

fn frame_to_tick(frame: usize, sample_rate: u32) -> u32 {
    // Assuming 120 BPM and 480 Ticks Per Quarter Note (TPQN)
    // 1 beat = 0.5s = 480 ticks -> 960 ticks per second
    ((frame as f64 / sample_rate as f64) * 960.0).round() as u32
}

fn main() {
    println!("Loading audio...");
    let mut reader = hound::WavReader::open("chant.wav").expect("Failed to open chant.wav");
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    
    // Normalize audio to f32 for filtering
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => reader.samples::<i16>()
            .map(|s| s.unwrap() as f32 / i16::MAX as f32)
            .collect(),
        hound::SampleFormat::Float => reader.samples::<f32>()
            .map(|s| s.unwrap())
            .collect(),
    };

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
        
        // threshold 0.15, certainty 0.20 are standard gating limits
        if let Some(pitch) = detector.get_pitch(&window, sample_rate as usize, 0.15, 0.20) {
            raw_pitches.push((i, hz_to_midi(pitch.frequency)));
        } else {
            raw_pitches.push((i, 0.0));
        }
    }

    println!("Applying Median Filter to smooth vibrato...");
    let median_window = 7; // Smooth over ~7 frames
    let mut smoothed_pitches = Vec::new();
    
    for i in 0..raw_pitches.len() {
        let start = i.saturating_sub(median_window / 2);
        let end = (i + median_window / 2).min(raw_pitches.len() - 1);
        let mut window: Vec<f64> = raw_pitches[start..=end].iter().map(|&(_, p)| p).collect();
        
        window.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_val = window[window.len() / 2];
        smoothed_pitches.push((raw_pitches[i].0, median_val));
    }

    println!("Segmenting Notes with Hysteresis (Debouncing)...");
    let min_stable_frames = 6; // Pitch must hold steady for ~70ms to register
    let mut midi_events = Vec::new();
    
    let mut current_note: Option<(u8, usize)> = None;
    let mut candidate_note = 0u8;
    let mut candidate_count = 0;

    for &(frame_idx, midi_float) in &smoothed_pitches {
        let pitch = if midi_float > 0.0 { midi_float.round() as u8 } else { 0 };

        if pitch == candidate_note {
            candidate_count += 1;
        } else {
            candidate_note = pitch;
            candidate_count = 1;
        }

        // Once a pitch holds for `min_stable_frames`, lock it in.
        if candidate_count == min_stable_frames {
            let actual_start_frame = frame_idx.saturating_sub(min_stable_frames * hop_size);

            match current_note {
                Some((curr_pitch, start_frame)) => {
                    if curr_pitch != candidate_note {
                        if curr_pitch != 0 {
                            midi_events.push((curr_pitch, start_frame, actual_start_frame));
                        }
                        if candidate_note != 0 {
                            current_note = Some((candidate_note, actual_start_frame));
                        } else {
                            current_note = None; // Transitioned to silence
                        }
                    }
                }
                None => {
                    if candidate_note != 0 {
                        current_note = Some((candidate_note, actual_start_frame));
                    }
                }
            }
        }
    }

    // Flush any lingering note at the end of the file
    if let Some((curr_pitch, start_frame)) = current_note {
        midi_events.push((curr_pitch, start_frame, filtered_samples.len()));
    }

    println!("Writing MIDI file...");
    let header = Header::new(Format::SingleTrack, Timing::Metrical(480.into()));
    let mut track = Track::new();
    let mut last_tick = 0;

    for (pitch, start_frame, end_frame) in midi_events {
        let start_tick = frame_to_tick(start_frame, sample_rate);
        let end_tick = frame_to_tick(end_frame, sample_rate);

        let delta_on = start_tick.saturating_sub(last_tick);
        track.push(TrackEvent {
            delta: delta_on.into(),
            kind: TrackEventKind::Midi {
                channel: 0.into(),
                message: MidiMessage::NoteOn { 
                    key: u7::from(pitch), 
                    vel: u7::from(100) 
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

    let smf = Smf::new(header, vec![track]);
    smf.save("chant_output.mid").unwrap();
    println!("Done! Saved to chant_output.mid");
}
