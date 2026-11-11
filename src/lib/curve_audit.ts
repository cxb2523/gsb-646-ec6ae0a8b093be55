import { interpolateKeyframes, sampleAt } from './interpolate';
import { movesOf, type Move } from './moves';
import {
	cameraOf,
	departureOf,
	holdOf,
	MAX_CAMERA_LATITUDE,
	type CameraState,
	type Keyframe,
	type MapProjection
} from './types';

/**
 * A read-only explanation of why the camera is where it is at one moment of
 * an animation. Everything here is plain data: no functions, no class
 * instances, nothing from MapLibre or the DOM, so an audit round-trips
 * through `JSON.stringify`/`JSON.parse`. The module never touches the disk
 * and never mutates the keyframe array it is given.
 *
 * `camera` is always produced by `sampleAt` itself, so for the same inputs
 * it matches a direct `sampleAt(keyframes, t, projection)` field by field;
 * `moves` is the literal result of `movesOf`.
 */

/** Where in the timeline the sampled moment falls. */
export type AuditPhase =
	| 'before-first'
	| 'after-last'
	| 'holding'
	| 'moving'
	| 'single-keyframe'
	| 'invalid-time'
	| 'no-keyframes';

/** The geometry of the move the moment belongs to — mirrors `Move['kind']`. */
export type SegmentKind = 'arc' | 'linear' | 'curve';

/** Stable names of the checks, in the order they appear in `checks`. */
export type AuditCheckName =
	| 'keyframes-present'
	| 'time-finite'
	| 'range-hold'
	| 'stop-hold'
	| 'hold-fits-segment'
	| 'waypoint-velocity-continuity'
	| 'pitch-clamped'
	| 'latitude-clamped'
	| 'camera-finite';

export interface AuditCheck {
	name: AuditCheckName;
	passed: boolean;
	/** Why the check failed, or a note on what it observed; null for an ordinary pass. */
	reason: string | null;
}

/** The stop the camera rests on during a hold, when the phase is `holding`. */
export interface AuditHold {
	/** Index of the stop being held. */
	keyframeIndex: number;
	/** Time the camera arrived at the stop. */
	from: number;
	/** Time the camera leaves again. */
	until: number;
	/** Length of the hold in seconds. */
	duration: number;
}

/** The neighbours the curve of a waypoint segment was built from. */
export interface AuditNeighbours {
	prev: number | null;
	prev2: number | null;
	next: number | null;
	next2: number | null;
}

/** The segment (`keyframes[i]` → `keyframes[i + 1]`) the moment is sampled on. */
export interface AuditSegment {
	fromIndex: number;
	toIndex: number;
	/** Departure time of the origin, i.e. after its hold. */
	fromTime: number;
	toTime: number;
	/** 0…1 position of `t` between departure and arrival. */
	alpha: number;
	kind: SegmentKind;
	neighbours: AuditNeighbours;
}

export interface CurveAudit {
	/** The queried time, or null when it was not a finite number (JSON has no NaN). */
	t: number | null;
	projection: MapProjection;
	/** The move partition, exactly as `movesOf(keyframes)` returns it. */
	moves: Move[];
	/** The move containing the moment; null outside the defined range or without keyframes. */
	activeMove: Move | null;
	phase: AuditPhase;
	/** Set while the phase is `holding`. */
	hold: AuditHold | null;
	/** Set while the phase is `moving`. */
	segment: AuditSegment | null;
	/** The camera at `t`; null only for an empty keyframe array. */
	camera: CameraState | null;
	checks: AuditCheck[];
}

/** Half-width of the finite-difference window used for the velocity check, in seconds. */
const VELOCITY_EPSILON = 1e-4;
/** Floating-point tolerance shared with the callers' 1e-9 comparisons. */
const NUMERIC_EPSILON = 1e-9;
/** A slope smaller than this many units per second counts as "at rest". */
const REST_SLOPE = 1e-6;
/** Two slopes count as equal within a factor of 1.5 plus the rest floor. */
const SLOPE_RATIO_LIMIT = 1.5;

const CAMERA_KEYS = ['lng', 'lat', 'zoom', 'pitch', 'bearing', 'roll'] as const;
/** Components that wrap around the compass, so their differences take the short way. */
const ANGLE_KEYS = new Set<(typeof CAMERA_KEYS)[number]>(['lng', 'bearing', 'roll']);

function check(name: AuditCheckName, passed: boolean, reason: string | null = null): AuditCheck {
	return { name, passed, reason };
}

/** Shortest signed difference b − a for an angle in degrees. */
function angleDelta(a: number, b: number): number {
	return ((((b - a + 180) % 360) + 360) % 360) - 180;
}

function deltaOf(key: (typeof CAMERA_KEYS)[number], a: number, b: number): number {
	return ANGLE_KEYS.has(key) ? angleDelta(a, b) : b - a;
}

/**
 * A copy of the segment selection inside `sampleAt`, re-walked here so the
 * audit can report the indices and neighbours instead of only the camera.
 * Returns null when the moment is outside every segment (held or out of
 * range). It deliberately follows the same branch order as `sampleAt`.
 */
function locateSegment(
	keyframes: readonly Keyframe[],
	t: number
): { index: number; from: number; span: number; alpha: number } | null {
	for (let i = 0; i < keyframes.length - 1; i++) {
		const a = keyframes[i];
		const b = keyframes[i + 1];
		if (t >= a.t && t <= b.t) {
			const from = departureOf(a);
			if (t <= from) return null; // still on the stop's hold
			const span = b.t - from;
			return { index: i, from, span, alpha: span > 0 ? (t - from) / span : 0 };
		}
	}
	return null;
}

/**
 * The neighbours the spline for segment `i` depends on, mirroring
 * `sampleAt` exactly (including the `leaving(...)` time shift, which only
 * changes `.t` copies and therefore not the indices reported here).
 */
function neighboursOf(keyframes: readonly Keyframe[], i: number): AuditNeighbours {
	const a = keyframes[i];
	const b = keyframes[i + 1];
	const prev = a.waypoint && i > 0 ? i - 1 : null;
	const prev2 = prev !== null && keyframes[prev].waypoint && i > 1 ? i - 2 : null;
	const next = b.waypoint && i + 2 < keyframes.length ? i + 2 : null;
	const next2 =
		next !== null && keyframes[next].waypoint && i + 3 < keyframes.length ? i + 3 : null;
	return { prev, prev2, next, next2 };
}

/** Forward and backward finite-difference velocities of every camera component. */
function slopeVectors(
	keyframes: readonly Keyframe[],
	t: number,
	center: CameraState,
	projection: MapProjection
): { before: CameraState; center: CameraState; after: CameraState } | null {
	const before = sampleAt([...keyframes], t - VELOCITY_EPSILON, projection);
	const after = sampleAt([...keyframes], t + VELOCITY_EPSILON, projection);
	if (!before || !after) return null;
	return { before, center, after };
}

/**
 * Compare the velocity just before and just after a waypoint. The spline is
 * built to share its tangent across a waypoint, so the two finite
 * differences must agree within numerical noise. Angles are compared the
 * short way, so crossing the antimeridian does not look like a 360° jump.
 */
function velocityContinuityCheck(
	keyframes: readonly Keyframe[],
	t: number,
	center: CameraState,
	segment: AuditSegment | null,
	phase: AuditPhase,
	projection: MapProjection
): AuditCheck {
	const name = 'waypoint-velocity-continuity' satisfies AuditCheckName;
	if (phase === 'no-keyframes' || phase === 'invalid-time') {
		return check(
			name,
			false,
			'Cannot evaluate velocity: no finite moment inside an animation was given.'
		);
	}
	if (!segment || segment.kind !== 'curve') {
		return check(name, true, 'Not on a waypoint segment: nothing whose two sides could disagree.');
	}
	// A moment resting on a stop has zero velocity on both sides by construction.
	if (phase === 'holding') {
		return check(name, true, 'Held on a stop: velocity is zero on both sides.');
	}

	const atWaypoint = keyframes.findIndex((kf) => Math.abs(kf.t - t) <= VELOCITY_EPSILON);
	const boundaryNote =
		atWaypoint === 0
			? ' The first keyframe is a waypoint, so the animation starts in flight and the camera jumps from rest into motion.'
			: atWaypoint === keyframes.length - 1
				? ' The last keyframe is a waypoint, so the animation ends in flight: motion is cut off at the last frame.'
				: '';

	// At the boundary frames one side of the finite difference is the held
	// end frame, which never moves. A stop rests there anyway; only a
	// boundary waypoint — a start or end in flight — creates a real jump.
	if (atWaypoint === 0 || atWaypoint === keyframes.length - 1) {
		const boundary = keyframes[atWaypoint];
		if (boundary.waypoint) {
			return check(
				name,
				false,
				`Velocity is discontinuous across the ${atWaypoint === 0 ? 'first' : 'last'} keyframe at t=${boundary.t}: the held frame outside the animation is stationary but the spline moves.${boundaryNote}`
			);
		}
		return check(name, true, 'The boundary frame is a stop, where the camera comes to rest.');
	}

	const vectors = slopeVectors(keyframes, t, center, projection);
	if (!vectors) return check(name, false, 'Could not sample the moments around t.');
	const { before, center: at, after } = vectors;

	const mismatches: string[] = [];
	for (const key of CAMERA_KEYS) {
		const left = deltaOf(key, before[key], at[key]) / VELOCITY_EPSILON;
		const right = deltaOf(key, at[key], after[key]) / VELOCITY_EPSILON;
		const atRest = Math.abs(left) < REST_SLOPE && Math.abs(right) < REST_SLOPE;
		const close =
			Math.abs(left - right) <= REST_SLOPE ||
			(Math.abs(left) >= REST_SLOPE &&
				Math.abs(right) >= REST_SLOPE &&
				Math.max(Math.abs(left), Math.abs(right)) / Math.min(Math.abs(left), Math.abs(right)) <=
					SLOPE_RATIO_LIMIT);
		if (!atRest && !close) {
			mismatches.push(`${key}: ${left.toExponential(3)} vs ${right.toExponential(3)} per second`);
		}
	}

	if (mismatches.length === 0) {
		return check(
			name,
			true,
			'Finite-difference velocities on both sides of the moment agree within tolerance.'
		);
	}
	return check(
		name,
		false,
		`Velocity is discontinuous across the waypoint at t=${t} — ${mismatches.join('; ')}.${boundaryNote}`
	);
}

/**
 * Audit the camera trajectory of `keyframes` at time `t`.
 *
 * Pure: the keyframe array is only read, no MapLibre/DOM/disk APIs are used,
 * and the result is JSON-serializable. On any degenerate input (empty array,
 * one keyframe, NaN time, a hold longer than the gap to the next frame) it
 * returns a structured result with a readable reason instead of throwing.
 */
export function auditCurve(
	keyframes: readonly Keyframe[],
	t: number,
	projection: MapProjection = 'mercator'
): CurveAudit {
	const moves = movesOf(keyframes);
	const finiteTime = typeof t === 'number' && Number.isFinite(t);
	const checks: AuditCheck[] = [];

	// --- Empty animation -----------------------------------------------------
	if (keyframes.length === 0) {
		checks.push(
			check(
				'keyframes-present',
				false,
				'The keyframe array is empty: there is no camera to sample.'
			)
		);
		checks.push(
			check(
				'time-finite',
				finiteTime,
				finiteTime ? null : 't is not a finite number (NaN or infinity).'
			)
		);
		for (const [name, note] of [
			['range-hold', 'No range to clamp: there are no keyframes.'],
			['stop-hold', 'No stop can be held: there are no keyframes.'],
			['hold-fits-segment', 'No segment exists whose length a hold could exceed.'],
			['waypoint-velocity-continuity', 'Cannot evaluate velocity without keyframes.'],
			['pitch-clamped', 'No camera state was produced.'],
			['latitude-clamped', 'No camera state was produced.'],
			['camera-finite', 'No camera state was produced.']
		] as const) {
			checks.push(check(name, false, note));
		}
		return {
			t: finiteTime ? t : null,
			projection,
			moves,
			activeMove: null,
			phase: 'no-keyframes',
			hold: null,
			segment: null,
			camera: null,
			checks
		};
	}

	checks.push(check('keyframes-present', true, null));
	checks.push(
		check(
			'time-finite',
			finiteTime,
			finiteTime
				? null
				: 't is not a finite number (NaN or infinity); sampling falls back to the last keyframe.'
		)
	);

	// A single keyframe is held forever, the same as sampleAt's behaviour.
	if (keyframes.length === 1) {
		const only = keyframes[0];
		const camera = cameraOf(only);
		checks.push(
			check('range-hold', true, 'Only one keyframe: its view is held for the whole timeline.')
		);
		checks.push(
			check(
				'stop-hold',
				true,
				`Resting on the single keyframe at index 0; there is no later frame to leave for.`
			)
		);
		checks.push(check('hold-fits-segment', true, 'There is no following segment.'));
		checks.push(
			check(
				'waypoint-velocity-continuity',
				true,
				'A single keyframe never moves, so no velocity can change.'
			)
		);
		checks.push(...clampChecks(camera));
		return {
			t: finiteTime ? t : null,
			projection,
			moves,
			activeMove: null,
			phase: 'single-keyframe',
			hold: null,
			segment: null,
			camera,
			checks
		};
	}

	const first = keyframes[0];
	const last = keyframes[keyframes.length - 1];

	// --- Strictly outside the defined range: hold the end frame -------------
	// The boundary frames themselves (t === first.t / last.t) fall through to
	// the segment logic, so a waypoint that starts or ends the animation in
	// flight still gets its velocity checked. Match sampleAt's comparisons:
	// -Infinity holds the first frame, NaN and +Infinity hold the last one.
	if (!finiteTime || t < first.t || t > last.t) {
		const holdsFirst = finiteTime ? t < first.t : t === Number.NEGATIVE_INFINITY;
		const camera = holdsFirst ? cameraOf(first) : cameraOf(last);
		const phase: AuditPhase = finiteTime
			? holdsFirst
				? 'before-first'
				: 'after-last'
			: 'invalid-time';
		checks.push(
			check(
				'range-hold',
				true,
				finiteTime
					? holdsFirst
						? `t=${t} is before the first keyframe (t=${first.t}): the first frame is held.`
						: `t=${t} is after the last keyframe (t=${last.t}): the last frame is held.`
					: 't is not finite: sampleAt holds the last frame.'
			)
		);
		checks.push(check('stop-hold', true, 'No stop hold is active outside the keyframe range.'));
		checks.push(check('hold-fits-segment', true, 'No segment is being sampled.'));
		checks.push(
			check(
				'waypoint-velocity-continuity',
				true,
				'Outside the animated range the camera is stationary.'
			)
		);
		checks.push(...clampChecks(camera));
		return {
			t: finiteTime ? t : null,
			projection,
			moves,
			activeMove: null,
			phase,
			hold: null,
			segment: null,
			camera,
			checks
		};
	}

	// --- Inside the range: a hold or a segment ------------------------------
	const located = locateSegment(keyframes, t);

	if (!located) {
		// sampleAt returns the view of the stop being held. Find which one.
		const stopIndex = keyframes.findIndex(
			(kf) => t >= kf.t && t <= departureOf(kf) && holdOf(kf) > 0
		);
		const stop = stopIndex >= 0 ? keyframes[stopIndex] : null;
		checks.push(check('range-hold', true, 't is inside the keyframe range.'));

		// No held stop means this is a boundary frame at its own time. A
		// boundary waypoint starts or ends in flight, so it is a moving moment
		// whose one-sided velocity the check has to judge; a boundary stop rests.
		const boundaryIndex =
			stop === null ? keyframes.findIndex((kf) => Math.abs(kf.t - t) <= VELOCITY_EPSILON) : -1;
		if (boundaryIndex === 0 || boundaryIndex === keyframes.length - 1) {
			const i = boundaryIndex === 0 ? 0 : keyframes.length - 2;
			const a = keyframes[i];
			const b = keyframes[i + 1];
			const activeMove = moves.find((move) => i >= move.start && i < move.end) ?? null;
			const segment: AuditSegment = {
				fromIndex: i,
				toIndex: i + 1,
				fromTime: departureOf(a),
				toTime: b.t,
				alpha: boundaryIndex === 0 ? 0 : 1,
				kind: activeMove?.kind ?? a.path ?? 'arc',
				neighbours: neighboursOf(keyframes, i)
			};
			const camera = sampleAt([...keyframes], t, projection)!;
			checks.push(
				check(
					'stop-hold',
					true,
					`At the ${boundaryIndex === 0 ? 'first' : 'last'} keyframe's own time; no hold is active.`
				)
			);
			checks.push(globalHoldFitCheck(keyframes, i));
			checks.push(velocityContinuityCheck(keyframes, t, camera, segment, 'moving', projection));
			checks.push(...clampChecks(camera));
			return {
				t,
				projection,
				moves,
				activeMove,
				phase: 'moving',
				hold: null,
				segment,
				camera,
				checks
			};
		}

		const camera = stop ? cameraOf(stop) : sampleAt([...keyframes], t, projection)!;
		const until = stop ? departureOf(stop) : t;
		checks.push(
			check(
				'stop-hold',
				true,
				stop
					? `Stationary on stop #${stopIndex} from t=${stop.t} until it leaves at t=${until}.`
					: 'Sampling the keyframe at its own time.'
			)
		);
		checks.push(globalHoldFitCheck(keyframes, null));
		checks.push(
			check('waypoint-velocity-continuity', true, 'Held on a stop: velocity is zero on both sides.')
		);
		checks.push(...clampChecks(camera));
		return {
			t,
			projection,
			moves,
			activeMove: null,
			phase: 'holding',
			hold: stop ? { keyframeIndex: stopIndex, from: stop.t, until, duration: holdOf(stop) } : null,
			segment: null,
			camera,
			checks
		};
	}

	const i = located.index;
	const a = keyframes[i];
	const b = keyframes[i + 1];
	const activeMove = moves.find((move) => i >= move.start && i < move.end) ?? null;
	const kind: SegmentKind = activeMove?.kind ?? a.path ?? 'arc';
	const neighbours = neighboursOf(keyframes, i);
	const segment: AuditSegment = {
		fromIndex: i,
		toIndex: i + 1,
		fromTime: located.from,
		toTime: b.t,
		alpha: located.alpha,
		kind,
		neighbours
	};
	const camera = sampleAt([...keyframes], t, projection)!;

	checks.push(check('range-hold', true, 't is inside the keyframe range.'));
	checks.push(
		check(
			'stop-hold',
			true,
			holdOf(a) > 0
				? `The hold of stop #${i} ended at t=${located.from}; the camera is moving now.`
				: 'No stop hold is active at this moment.'
		)
	);
	checks.push(globalHoldFitCheck(keyframes, i));
	checks.push(velocityContinuityCheck(keyframes, t, camera, segment, 'moving', projection));
	checks.push(...clampChecks(camera));

	return {
		t,
		projection,
		moves,
		activeMove,
		phase: 'moving',
		hold: null,
		segment,
		camera,
		checks
	};
}

/** The pitch/latitude/finiteness checks, run on whatever camera was produced. */
function clampChecks(camera: CameraState): AuditCheck[] {
	const pitchOk = camera.pitch >= -NUMERIC_EPSILON && camera.pitch <= 90 + NUMERIC_EPSILON;
	const latOk = Math.abs(camera.lat) <= MAX_CAMERA_LATITUDE + NUMERIC_EPSILON;
	const finite = CAMERA_KEYS.every((key) => Number.isFinite(camera[key]));
	return [
		check(
			'pitch-clamped',
			pitchOk,
			pitchOk ? null : `pitch ${camera.pitch} is outside the allowed 0…90 range.`
		),
		check(
			'latitude-clamped',
			latOk,
			latOk ? null : `latitude ${camera.lat} exceeds MAX_CAMERA_LATITUDE (±${MAX_CAMERA_LATITUDE}).`
		),
		check(
			'camera-finite',
			finite,
			finite ? null : 'At least one camera component is NaN or infinite.'
		)
	];
}

/** The check against one concrete stop, shared by the two call sites. */
function holdFitResult(
	name: AuditCheckName,
	stop: Keyframe,
	stopIndex: number,
	next?: Keyframe
): AuditCheck {
	const following = next ?? null;
	if (!following) return check(name, true, 'No following segment for this hold to overflow into.');
	const gap = following.t - stop.t;
	const hold = holdOf(stop);
	if (hold <= gap) {
		return check(name, true, `The hold of ${hold}s fits inside the ${gap}s gap to the next frame.`);
	}
	return check(
		name,
		false,
		`The hold on stop #${stopIndex} is ${hold}s but the next frame is only ${gap}s away: the move has no time left and the departure lands at or past the next frame.`
	);
}

/**
 * Any stop whose hold reaches the next frame makes that segment degenerate.
 * The check is run for every audit moment, so the problem stays visible
 * while the zero-length move itself is being sampled. When a segment is
 * actively sampled its own stop is reported first.
 */
function globalHoldFitCheck(
	keyframes: readonly Keyframe[],
	segmentIndex: number | null
): AuditCheck {
	const name = 'hold-fits-segment' satisfies AuditCheckName;
	if (segmentIndex !== null) {
		const own = holdFitResult(
			name,
			keyframes[segmentIndex],
			segmentIndex,
			keyframes[segmentIndex + 1]
		);
		if (!own.passed) return own;
	}
	for (let i = 0; i < keyframes.length - 1; i++) {
		const result = holdFitResult(name, keyframes[i], i, keyframes[i + 1]);
		if (!result.passed) return result;
	}
	return check(name, true, 'Every hold leaves time for the move that follows its stop.');
}

/**
 * Re-run the exact interpolation the segment selection would make, with the
 * same neighbour indices the audit reports. Exposed for tests that want to
 * prove the reported neighbours are the ones actually used; production code
 * gets the camera from `sampleAt` via `auditCurve`.
 */
export function auditedInterpolation(
	keyframes: readonly Keyframe[],
	segment: AuditSegment,
	projection: MapProjection
): CameraState {
	const i = segment.fromIndex;
	const leaving = (kf: Keyframe): Keyframe => ({ ...kf, t: departureOf(kf) });
	const { prev, prev2, next, next2 } = segment.neighbours;
	return interpolateKeyframes(leaving(keyframes[i]), keyframes[i + 1], segment.alpha, projection, {
		prev: prev === null ? null : leaving(keyframes[prev]),
		prev2: prev2 === null ? null : leaving(keyframes[prev2]),
		next: next === null ? null : keyframes[next],
		next2: next2 === null ? null : keyframes[next2]
	});
}
