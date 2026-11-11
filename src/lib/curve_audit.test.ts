import { describe, expect, it } from 'vitest';
import { auditCurve, auditedInterpolation, type CurveAudit } from './curve_audit';
import { sampleAt } from './interpolate';
import { movesOf } from './moves';
import { MAX_CAMERA_LATITUDE, type CameraState, type Keyframe, type MapProjection } from './types';

const CAMERA_KEYS = ['lng', 'lat', 'zoom', 'pitch', 'bearing', 'roll'] as const;

const stop = (t: number, lng: number, extra: Partial<Keyframe> = {}): Keyframe => ({
	t,
	lng,
	lat: 0,
	zoom: 5,
	pitch: 0,
	bearing: 0,
	roll: 0,
	...extra
});
const via = (t: number, lng: number, extra: Partial<Keyframe> = {}): Keyframe =>
	stop(t, lng, { waypoint: true, ...extra });

/** Field-by-field comparison with the 1e-9 tolerance the contract asks for. */
function expectCameraEqual(actual: CameraState | null, expected: CameraState | null): void {
	if (expected === null || actual === null) {
		expect(actual).toBe(expected);
		return;
	}
	for (const key of CAMERA_KEYS) expect(actual[key], key).toBeCloseTo(expected[key], 9);
}

function checkOf(audit: CurveAudit, name: string) {
	const found = audit.checks.find((c) => c.name === name);
	if (!found)
		throw new Error(`check ${name} missing; got ${audit.checks.map((c) => c.name).join(', ')}`);
	return found;
}

describe('auditCurve — mixed sequence', () => {
	// One held stop, a move through two waypoints, a plain linear move and a
	// default arc move, spanning both projections.
	const keyframes: Keyframe[] = [
		stop(0, 0, { hold: 1 }),
		via(2, 10),
		via(3, 20),
		stop(4, 30, { path: 'linear' }),
		stop(6, 40),
		stop(8, 50)
	];
	const snapshot = keyframes.map((kf) => ({ ...kf }));

	const expectedMoves = [
		{ start: 0, end: 3, kind: 'curve' },
		{ start: 3, end: 4, kind: 'linear' },
		{ start: 4, end: 5, kind: 'arc' }
	] as const;

	it('partitions the moves exactly like movesOf', () => {
		const audit = auditCurve(keyframes, 2.5);
		expect(audit.moves).toEqual(movesOf(keyframes));
		expect(audit.moves).toEqual(expectedMoves);
	});

	it('holds the first frame before the first keyframe', () => {
		const audit = auditCurve(keyframes, -3);
		expect(audit.phase).toBe('before-first');
		expect(audit.activeMove).toBeNull();
		expect(audit.segment).toBeNull();
		expect(audit.hold).toBeNull();
		expectCameraEqual(audit.camera, sampleAt(keyframes, -3));
		expect(checkOf(audit, 'range-hold').passed).toBe(true);
		expect(checkOf(audit, 'range-hold').reason).toContain('first frame is held');
	});

	it('holds the last frame after the last keyframe', () => {
		const audit = auditCurve(keyframes, 42);
		expect(audit.phase).toBe('after-last');
		expectCameraEqual(audit.camera, sampleAt(keyframes, 42));
		expect(checkOf(audit, 'range-hold').reason).toContain('last frame is held');
	});

	it('reports a stationary stop with its departure time during a hold', () => {
		const audit = auditCurve(keyframes, 0.5);
		expect(audit.phase).toBe('holding');
		expect(audit.hold).toEqual({ keyframeIndex: 0, from: 0, until: 1, duration: 1 });
		expect(audit.activeMove).toBeNull();
		const stopCheck = checkOf(audit, 'stop-hold');
		expect(stopCheck.passed).toBe(true);
		expect(stopCheck.reason).toContain('Stationary on stop #0');
		expect(stopCheck.reason).toContain('leaves at t=1');
		expectCameraEqual(audit.camera, sampleAt(keyframes, 0.5));
	});

	it('audits a curve segment mid-way with full neighbour chains', () => {
		const audit = auditCurve(keyframes, 2.5);
		expect(audit.phase).toBe('moving');
		expect(audit.activeMove).toEqual({ start: 0, end: 3, kind: 'curve' });
		expect(audit.segment).toMatchObject({
			fromIndex: 1,
			toIndex: 2,
			kind: 'curve',
			neighbours: { prev: 0, prev2: null, next: 3, next2: null }
		});
		expect(audit.segment?.alpha).toBeCloseTo(0.5, 9);
		expectCameraEqual(audit.camera, sampleAt(keyframes, 2.5));
		// The reported neighbours really are the ones the interpolation uses.
		expectCameraEqual(
			auditedInterpolation(keyframes, audit.segment!, 'mercator'),
			sampleAt(keyframes, 2.5)
		);
	});

	it('fills prev2/next2 two waypoints out', () => {
		const dense = [stop(0, 0), via(1, 10), via(2, 20), via(3, 30), via(4, 40), stop(5, 50)];
		// t=2.5 is on segment 2→3, with two waypoints on either side.
		const audit = auditCurve(dense, 2.5);
		expect(audit.segment?.fromIndex).toBe(2);
		expect(audit.segment?.neighbours).toEqual({ prev: 1, prev2: 0, next: 4, next2: 5 });
		expectCameraEqual(audit.camera, sampleAt(dense, 2.5));
	});

	it('ends the neighbour chain at the next stop', () => {
		const dense = [stop(0, 0), via(1, 10), via(2, 20), via(3, 30), stop(4, 40)];
		// Segment 2→3: kf[4] is a stop, so the chain ends there.
		const audit = auditCurve(dense, 2.5);
		expect(audit.segment?.fromIndex).toBe(2);
		expect(audit.segment?.neighbours).toEqual({ prev: 1, prev2: 0, next: 4, next2: null });
		expectCameraEqual(audit.camera, sampleAt(dense, 2.5));
	});

	it('reports null neighbours at the ends of the chain', () => {
		const audit = auditCurve(keyframes, 1.5); // segment 0 → 1
		expect(audit.segment?.fromIndex).toBe(0);
		// Nothing before the first keyframe; the two following frames are waypoints.
		expect(audit.segment?.neighbours).toEqual({ prev: null, prev2: null, next: 2, next2: 3 });
		expectCameraEqual(audit.camera, sampleAt(keyframes, 1.5));
	});

	it('audits linear and arc segments with the right kind', () => {
		const linear = auditCurve(keyframes, 5);
		expect(linear.segment).toMatchObject({ fromIndex: 3, toIndex: 4, kind: 'linear' });
		expect(linear.segment?.neighbours).toEqual({
			prev: null,
			prev2: null,
			next: null,
			next2: null
		});
		expectCameraEqual(linear.camera, sampleAt(keyframes, 5));

		const arc = auditCurve(keyframes, 7);
		expect(arc.segment).toMatchObject({ fromIndex: 4, toIndex: 5, kind: 'arc' });
		expectCameraEqual(arc.camera, sampleAt(keyframes, 7));
	});

	it('matches sampleAt at every sampled moment, including keyframe times', () => {
		for (const t of [-1, 0, 0.9, 1, 1.01, 2, 2.5, 3, 3.7, 4, 5, 6, 7.3, 8, 99]) {
			const audit = auditCurve(keyframes, t);
			expectCameraEqual(audit.camera, sampleAt(keyframes, t));
		}
	});

	it('matches sampleAt on the globe projection as well', () => {
		for (const projection of ['mercator', 'globe'] as MapProjection[]) {
			for (const t of [-1, 0.5, 1.5, 2, 2.5, 4, 5, 8, 12]) {
				const audit = auditCurve(keyframes, t, projection);
				expect(audit.projection).toBe(projection);
				expectCameraEqual(audit.camera, sampleAt(keyframes, t, projection));
			}
		}
	});

	it('never mutates the keyframe array', () => {
		auditCurve(keyframes, 2.5);
		auditCurve(keyframes, 0.5);
		expect(keyframes).toEqual(snapshot);
	});
});

describe('auditCurve — velocity continuity across a waypoint', () => {
	const smooth = [stop(0, 0), via(1, 10), via(2, 20), stop(3, 30)];

	it('passes on an ordinary spline through interior waypoints', () => {
		// The check under its real name, at a waypoint and between waypoints.
		expect(checkOf(auditCurve(smooth, 1), 'waypoint-velocity-continuity').passed).toBe(true);
		expect(checkOf(auditCurve(smooth, 2), 'waypoint-velocity-continuity').passed).toBe(true);
		expect(checkOf(auditCurve(smooth, 1.5), 'waypoint-velocity-continuity').passed).toBe(true);
	});

	it('reports a failure with a reason when the animation starts in flight at a waypoint', () => {
		// The first keyframe is a waypoint: before t=0 the camera is held still,
		// then it jumps into the motion of the spline — a genuine one-sided
		// velocity discontinuity at the boundary frame.
		const flyingStart: Keyframe[] = [via(0, 0), via(1, 10), stop(2, 20)];
		const audit = auditCurve(flyingStart, 0);
		const velocityCheck = checkOf(audit, 'waypoint-velocity-continuity');
		expect(velocityCheck.passed).toBe(false);
		expect(velocityCheck.reason).toContain('Velocity is discontinuous');
		expect(velocityCheck.reason).toContain('starts in flight');
		// The boundary waypoint is a moving moment on its real first segment.
		expect(audit.phase).toBe('moving');
		expect(audit.hold).toBeNull();
		expect(audit.segment).toMatchObject({ fromIndex: 0, toIndex: 1, kind: 'curve' });
		expect(audit.activeMove).toEqual({ start: 0, end: 2, kind: 'curve' });
		// The rest of the audit still agrees with sampleAt.
		expectCameraEqual(audit.camera, sampleAt(flyingStart, 0));
	});

	it('reports the mirrored failure when the animation ends in flight', () => {
		const flyingEnd: Keyframe[] = [stop(0, 0), via(1, 10), via(2, 20)];
		const audit = auditCurve(flyingEnd, 2);
		const velocityCheck = checkOf(audit, 'waypoint-velocity-continuity');
		expect(velocityCheck.passed).toBe(false);
		expect(velocityCheck.reason).toContain('ends in flight');
		// The last waypoint sits on the final segment, reported as moving.
		expect(audit.phase).toBe('moving');
		expect(audit.segment).toMatchObject({ fromIndex: 1, toIndex: 2, kind: 'curve' });
		expectCameraEqual(audit.camera, sampleAt(flyingEnd, 2));
	});

	it('treats a boundary stop at its own time as a rest, not a discontinuity', () => {
		const stopping: Keyframe[] = [stop(0, 0), via(1, 10), stop(2, 20)];
		const atEnd = auditCurve(stopping, 2);
		expect(checkOf(atEnd, 'waypoint-velocity-continuity').passed).toBe(true);
		expect(atEnd.phase).toBe('moving');
		expectCameraEqual(atEnd.camera, sampleAt(stopping, 2));

		// The same at the first frame: a plain stop resting there is not a jump.
		const plainStart: Keyframe[] = [stop(0, 0), stop(1, 10)];
		const atStart = auditCurve(plainStart, 0);
		expect(atStart.phase).toBe('moving');
		expect(atStart.segment).toMatchObject({ fromIndex: 0, toIndex: 1, kind: 'arc' });
		expect(checkOf(atStart, 'waypoint-velocity-continuity').passed).toBe(true);
		expect(checkOf(atStart, 'waypoint-velocity-continuity').reason).toContain(
			'Not on a waypoint segment'
		);
		expectCameraEqual(atStart.camera, sampleAt(plainStart, 0));
	});

	it('does not flag the same spline a moment inside the segment', () => {
		const flyingStart: Keyframe[] = [via(0, 0), via(1, 10), stop(2, 20)];
		expect(checkOf(auditCurve(flyingStart, 0.5), 'waypoint-velocity-continuity').passed).toBe(true);
	});

	it('skips the check with a note on non-curve segments', () => {
		const plain = [stop(0, 0, { path: 'linear' }), stop(2, 10)];
		const velocityCheck = checkOf(auditCurve(plain, 1), 'waypoint-velocity-continuity');
		expect(velocityCheck.passed).toBe(true);
		expect(velocityCheck.reason).toContain('Not on a waypoint segment');
	});

	it('takes the short way across the antimeridian instead of seeing a 360° jump', () => {
		const wrap = [
			stop(0, 170, { bearing: 170 }),
			via(1, -180, { bearing: -180 }),
			stop(2, -170, { bearing: -170 })
		];
		const audit = auditCurve(wrap, 1);
		expect(checkOf(audit, 'waypoint-velocity-continuity').passed).toBe(true);
		expectCameraEqual(audit.camera, sampleAt(wrap, 1));
	});
});

describe('auditCurve — clamp checks', () => {
	it('passes pitch and latitude checks for ordinary cameras', () => {
		const keyframes = [stop(0, 0), stop(2, 10, { pitch: 45, lat: 60 })];
		const audit = auditCurve(keyframes, 1);
		expect(checkOf(audit, 'pitch-clamped').passed).toBe(true);
		expect(checkOf(audit, 'latitude-clamped').passed).toBe(true);
		expect(checkOf(audit, 'camera-finite').passed).toBe(true);
	});

	it('the reported limits are the documented constants', () => {
		const polar = [stop(0, 10, { lat: 84 }), stop(2, -170, { lat: 84 })];
		const audit = auditCurve(polar, 1, 'globe');
		expect(Math.abs(audit.camera!.lat)).toBeLessThanOrEqual(MAX_CAMERA_LATITUDE + 1e-9);
		expect(checkOf(audit, 'latitude-clamped').passed).toBe(true);
	});
});

describe('auditCurve — hold longer than the gap', () => {
	it('flags the overlong hold but still returns the sampleAt camera without throwing', () => {
		// Hold of 5 s while the next frame is only 3 s away.
		const keyframes: Keyframe[] = [stop(0, 0, { hold: 5 }), stop(3, 10), stop(6, 20)];
		const audit = auditCurve(keyframes, 1);
		expect(audit.phase).toBe('holding');
		const fit = checkOf(audit, 'hold-fits-segment');
		expect(fit.passed).toBe(false);
		expect(fit.reason).toContain('5s');
		expect(fit.reason).toContain('3s');
		expectCameraEqual(audit.camera, sampleAt(keyframes, 1));

		// After the nominal departure the degenerate (zero-length) move is sampled.
		const later = auditCurve(keyframes, 4);
		expect(checkOf(later, 'hold-fits-segment').passed).toBe(false);
		expectCameraEqual(later.camera, sampleAt(keyframes, 4));
	});
});

describe('auditCurve — boundary inputs', () => {
	it('returns a structured failure for an empty array', () => {
		const audit = auditCurve([], 2);
		expect(audit.phase).toBe('no-keyframes');
		expect(audit.camera).toBeNull();
		expect(audit.moves).toEqual([]);
		expect(audit.activeMove).toBeNull();
		expect(audit.segment).toBeNull();
		expect(checkOf(audit, 'keyframes-present').passed).toBe(false);
		expect(checkOf(audit, 'keyframes-present').reason).toBeTypeOf('string');
		expect(audit.checks).toHaveLength(9);
		expect(audit.checks.filter((c) => !c.passed).every((c) => typeof c.reason === 'string')).toBe(
			true
		);
	});

	it('holds the only keyframe for a single-keyframe array', () => {
		const only = stop(7, 12);
		for (const t of [-10, 0, 7, 99]) {
			const audit = auditCurve([only], t);
			expect(audit.phase).toBe('single-keyframe');
			expect(audit.moves).toEqual([]);
			expectCameraEqual(audit.camera, sampleAt([only], t));
			expect(checkOf(audit, 'range-hold').reason).toContain('held for the whole timeline');
		}
	});

	it('handles NaN t with a readable result instead of throwing', () => {
		const keyframes = [stop(0, 0), via(1, 10), stop(2, 20)];
		const audit = auditCurve(keyframes, Number.NaN);
		expect(audit.t).toBeNull();
		expect(audit.phase).toBe('invalid-time');
		expect(checkOf(audit, 'time-finite').passed).toBe(false);
		expect(checkOf(audit, 'time-finite').reason).toContain('not a finite number');
		expectCameraEqual(audit.camera, sampleAt(keyframes, Number.NaN));
	});

	it('also survives infinite t and NaN t on empty and single-keyframe inputs', () => {
		expect(auditCurve([], Number.NaN).phase).toBe('no-keyframes');
		const single = auditCurve([stop(0, 0)], Number.POSITIVE_INFINITY);
		expect(single.phase).toBe('single-keyframe');
		expect(single.t).toBeNull();
		const pair: Keyframe[] = [stop(0, 0), stop(1, 1)];
		const tail = auditCurve(pair, Number.NEGATIVE_INFINITY);
		expect(tail.phase).toBe('invalid-time');
		// sampleAt treats -Infinity like "before the first frame" (first held),
		// while NaN and +Infinity hold the last frame.
		expectCameraEqual(tail.camera, sampleAt(pair, Number.NEGATIVE_INFINITY));
		expect(tail.camera?.lng).toBe(0);
		const plusInf = auditCurve(pair, Number.POSITIVE_INFINITY);
		expectCameraEqual(plusInf.camera, sampleAt(pair, Number.POSITIVE_INFINITY));
		expect(plusInf.camera?.lng).toBe(1);
	});

	it('does not throw on keyframes sharing a time', () => {
		const degenerate = [stop(0, 0), via(0, 5), stop(2, 10)];
		expect(() => auditCurve(degenerate, 1)).not.toThrow();
		const audit = auditCurve(degenerate, 1);
		expectCameraEqual(audit.camera, sampleAt(degenerate, 1));
	});
});

describe('auditCurve — JSON serializability and shape', () => {
	const keyframes: Keyframe[] = [
		stop(0, 0, { hold: 1 }),
		via(2, 10),
		stop(4, 30, { path: 'linear' })
	];

	it('stringifies and round-trips without losing data', () => {
		for (const t of [-1, 0.5, 2.5, 4, Number.NaN]) {
			const audit = auditCurve(keyframes, t);
			const text = JSON.stringify(audit);
			expect(typeof text).toBe('string');
			const back = JSON.parse(text) as CurveAudit;
			expect(back.phase).toBe(audit.phase);
			expect(back.checks.map((c) => c.name)).toEqual(audit.checks.map((c) => c.name));
			expect(back.checks.map((c) => c.passed)).toEqual(audit.checks.map((c) => c.passed));
			if (audit.camera) expectCameraEqual(back.camera as CameraState, audit.camera);
		}
	});

	it('contains no function-valued fields anywhere in the object', () => {
		const audit = auditCurve(keyframes, 2.5);
		const walk = (value: unknown): void => {
			if (value === null || value === undefined) return;
			expect(typeof value).not.toBe('function');
			if (typeof value === 'object') for (const child of Object.values(value)) walk(child);
		};
		walk(audit);
	});

	it('always emits the full ordered check list with readable reasons on failure', () => {
		const audit = auditCurve(keyframes, 2.5);
		expect(audit.checks.map((c) => c.name)).toEqual([
			'keyframes-present',
			'time-finite',
			'range-hold',
			'stop-hold',
			'hold-fits-segment',
			'waypoint-velocity-continuity',
			'pitch-clamped',
			'latitude-clamped',
			'camera-finite'
		]);
		for (const c of audit.checks) {
			expect(c.passed).toBeTypeOf('boolean');
			if (!c.passed) expect(c.reason).toBeTypeOf('string');
		}
	});
});
