/**
 * Zero-Dependency Institutional Web Audio API Synthesizer & Voice Callout Engine
 * (`ExecutionAudioEngine`)
 *
 * Uses pure browser `AudioContext` (`OscillatorNode`, `GainNode`, exponential envelope ramps)
 * and `window.speechSynthesis` (`SpeechSynthesisUtterance`) with zero external `.mp3`/`.wav` assets.
 * Deduplicates audio events across detached multi-monitor windows via `localStorage` (`argon-last-audio-event-id`).
 */

export type ExecutionAudioMode = 'VOICE_AND_CHIMES' | 'CHIMES_ONLY' | 'MUTED';

export const AUDIO_MODE_STORAGE_KEY = 'argon-execution-audio-mode';
export const LAST_AUDIO_EVENT_STORAGE_KEY = 'argon-last-audio-event-id';

export interface SynthesizedNote {
	freqHz: number;
	startOffsetSec: number;
	durationSec: number;
	type: OscillatorType;
	peakGain: number;
}

export class ExecutionAudioEngine {
	private ctx: AudioContext | null = null;
	private mode: ExecutionAudioMode = 'VOICE_AND_CHIMES';
	private lastSpokenText: string = '';
	private lastChimeSignature: string = '';

	constructor(initialMode?: ExecutionAudioMode) {
		if (initialMode) {
			this.mode = initialMode;
		} else if (typeof window !== 'undefined' && window.localStorage) {
			const saved = window.localStorage.getItem(AUDIO_MODE_STORAGE_KEY);
			if (saved === 'VOICE_AND_CHIMES' || saved === 'CHIMES_ONLY' || saved === 'MUTED') {
				this.mode = saved;
			}
		}
	}

	public getMode(): ExecutionAudioMode {
		return this.mode;
	}

	public setMode(nextMode: ExecutionAudioMode): void {
		this.mode = nextMode;
		if (typeof window !== 'undefined' && window.localStorage) {
			window.localStorage.setItem(AUDIO_MODE_STORAGE_KEY, nextMode);
		}
	}

	public cycleMode(): ExecutionAudioMode {
		const next: ExecutionAudioMode =
			this.mode === 'VOICE_AND_CHIMES'
				? 'CHIMES_ONLY'
				: this.mode === 'CHIMES_ONLY'
					? 'MUTED'
					: 'VOICE_AND_CHIMES';
		this.setMode(next);
		return next;
	}

	public getLastSpokenText(): string {
		return this.lastSpokenText;
	}

	public getLastChimeSignature(): string {
		return this.lastChimeSignature;
	}

	/**
	 * Cross-window deduplication using `localStorage` (`argon-last-audio-event-id`).
	 * Ensures that when 3 detached multi-monitor windows are open simultaneously,
	 * only the first window to process a trade event ID emits the chime/voice callout.
	 */
	public claimCrossWindowAudioEvent(eventId: string): boolean {
		if (!eventId) return false;
		if (typeof window === 'undefined' || !window.localStorage) {
			return true;
		}
		const prev = window.localStorage.getItem(LAST_AUDIO_EVENT_STORAGE_KEY);
		if (prev === eventId) {
			return false;
		}
		window.localStorage.setItem(LAST_AUDIO_EVENT_STORAGE_KEY, eventId);
		return true;
	}

	private ensureAudioContext(): AudioContext | null {
		if (typeof window === 'undefined') return null;
		const AudioCtx =
			window.AudioContext ||
			(window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
		if (!AudioCtx) return null;
		if (!this.ctx) {
			this.ctx = new AudioCtx();
		}
		if (this.ctx.state === 'suspended') {
			void this.ctx.resume().catch(() => {});
		}
		return this.ctx;
	}

	private scheduleNotes(notes: SynthesizedNote[], signatureLabel: string): void {
		this.lastChimeSignature = signatureLabel;
		if (this.mode === 'MUTED') return;
		const ctx = this.ensureAudioContext();
		if (!ctx) return;

		const now = ctx.currentTime;
		for (const note of notes) {
			const osc = ctx.createOscillator();
			const gain = ctx.createGain();
			osc.type = note.type;
			osc.frequency.setValueAtTime(note.freqHz, now + note.startOffsetSec);

			const start = now + note.startOffsetSec;
			const stop = start + note.durationSec;
			gain.gain.setValueAtTime(0.0001, start);
			gain.gain.exponentialRampToValueAtTime(Math.max(0.001, note.peakGain), start + 0.018);
			gain.gain.exponentialRampToValueAtTime(0.0001, stop);

			osc.connect(gain);
			gain.connect(ctx.destination);
			osc.start(start);
			osc.stop(stop + 0.01);
		}
	}

	/**
	 * Rising dual-tone harmonic chord (`523.25 Hz -> 659.25 Hz` C5-E5) for `LONG` entries;
	 * descending harmonic chord (`587.33 Hz -> 440.00 Hz` D5-A4) for `SHORT` entries.
	 */
	public playSignalEntryChime(direction: 'LONG' | 'SHORT'): void {
		if (direction === 'LONG') {
			this.scheduleNotes(
				[
					{
						freqHz: 523.25,
						startOffsetSec: 0,
						durationSec: 0.16,
						type: 'triangle',
						peakGain: 0.14
					},
					{
						freqHz: 659.25,
						startOffsetSec: 0.11,
						durationSec: 0.24,
						type: 'sine',
						peakGain: 0.16
					}
				],
				'ENTRY_LONG_523.25_659.25'
			);
		} else {
			this.scheduleNotes(
				[
					{
						freqHz: 587.33,
						startOffsetSec: 0,
						durationSec: 0.16,
						type: 'triangle',
						peakGain: 0.14
					},
					{
						freqHz: 440.0,
						startOffsetSec: 0.11,
						durationSec: 0.24,
						type: 'sine',
						peakGain: 0.16
					}
				],
				'ENTRY_SHORT_587.33_440.00'
			);
		}
	}

	/**
	 * Crisp high-frequency institutional fill ping (`880 Hz -> 1174.66 Hz` A5-D6).
	 */
	public playTarget1BankedChime(): void {
		this.scheduleNotes(
			[
				{
					freqHz: 880.0,
					startOffsetSec: 0,
					durationSec: 0.12,
					type: 'sine',
					peakGain: 0.14
				},
				{
					freqHz: 1174.66,
					startOffsetSec: 0.09,
					durationSec: 0.22,
					type: 'sine',
					peakGain: 0.15
				}
			],
			'TARGET1_BANKED_880.00_1174.66'
		);
	}

	/**
	 * Crisp high-frequency institutional scale-in fill ping (`880 Hz -> 1174.66 Hz` A5-D6).
	 */
	public playScaleInChime(): void {
		this.scheduleNotes(
			[
				{
					freqHz: 880.0,
					startOffsetSec: 0,
					durationSec: 0.11,
					type: 'triangle',
					peakGain: 0.13
				},
				{
					freqHz: 1174.66,
					startOffsetSec: 0.08,
					durationSec: 0.2,
					type: 'sine',
					peakGain: 0.15
				}
			],
			'SCALE_IN_880.00_1174.66'
		);
	}

	/**
	 * Major triad resolution chime for winning exits (`TARGET HIT` / positive `AI DYNAMIC EXIT`);
	 * muted low-tone alert (`220 Hz`) for `STOPPED OUT` / losing exits.
	 */
	public playPositionClosedChime(isWinner: boolean): void {
		if (isWinner) {
			this.scheduleNotes(
				[
					{
						freqHz: 523.25,
						startOffsetSec: 0,
						durationSec: 0.13,
						type: 'sine',
						peakGain: 0.13
					},
					{
						freqHz: 659.25,
						startOffsetSec: 0.09,
						durationSec: 0.15,
						type: 'sine',
						peakGain: 0.14
					},
					{
						freqHz: 783.99,
						startOffsetSec: 0.18,
						durationSec: 0.28,
						type: 'sine',
						peakGain: 0.16
					}
				],
				'CLOSED_WINNER_MAJOR_TRIAD'
			);
		} else {
			this.scheduleNotes(
				[
					{
						freqHz: 220.0,
						startOffsetSec: 0,
						durationSec: 0.26,
						type: 'sawtooth',
						peakGain: 0.09
					}
				],
				'CLOSED_STOPPED_220.00'
			);
		}
	}

	/**
	 * Institutional voice callout via `window.speechSynthesis` (`SpeechSynthesisUtterance`)
	 * with rate `1.08` and pitch `0.98`. Only active in `VOICE_AND_CHIMES` mode.
	 */
	public speakExecutionCallout(text: string): void {
		this.lastSpokenText = text;
		if (this.mode !== 'VOICE_AND_CHIMES') return;
		if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;
		try {
			window.speechSynthesis.cancel();
			const utterance = new SpeechSynthesisUtterance(text);
			utterance.rate = 1.08;
			utterance.pitch = 0.98;
			utterance.volume = 0.92;
			window.speechSynthesis.speak(utterance);
		} catch {
			// Ignore browser speech synthesis gesture errors
		}
	}

	/**
	 * Triggered by `[ Test Sound ]` button to unlock browser `AudioContext` gesture policies cleanly.
	 */
	public playTestSound(): void {
		if (this.mode === 'MUTED') {
			this.setMode('VOICE_AND_CHIMES');
		}
		this.playSignalEntryChime('LONG');
		if (this.mode === 'VOICE_AND_CHIMES') {
			const spelled = 'N V D A';
			this.speakExecutionCallout(
				`Argon Audio Synthesizer active. Brain 2 Long ${spelled} telemetry online.`
			);
		}
	}

	public static formatSymbolForSpeech(symbol: string): string {
		return symbol.trim().toUpperCase().split('').join(' ');
	}
}

export const executionAudio = new ExecutionAudioEngine();

