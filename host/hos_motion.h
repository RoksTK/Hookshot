/* (hookshot) A controller's motion sensors as the Switch reports its six-axis state (host/pad_sdl.c fills it from
   SDL, host/nn_hid.c hands it to the game's GetSixAxisSensorStates). Header-only, no SDL or Windows, so
   tests/test_motion.py runs it as it is.

   This is Eden's (and yuzu's) motion, the emulators OoT aims right with: SDL's readings turned into the Switch's frame
   and units, and the state built as the console builds it, so the game gets what it gets on hardware and does its own
   aiming (it recenters on the camera each time it aims; how the pad is held, in bed or at a desk, does not matter):
   - acceleration in G and angular velocity in turns a second, in the Switch's frame (Eden's sdl_driver.cpp:
     x = -X, y = Z, z = -Y for the accelerometer, x = X, y = -Z, z = Y for the gyroscope, SDL's +X right, +Y out of the
     face buttons, +Z toward the player), so a pad lying face up reads (0, 0, -1) G;
   - the rotation: the angular velocity summed (turns);
   - the direction: the pad's orientation as a 3x3 matrix, from Mahony's filter (the gyroscope integrated as a
     quaternion, pulled toward the measured gravity: kp 0.3, ki 0.005), Eden's motion_input.cpp line for line.
   The gyroscope's resting rate (its bias) is learned while the pad is still on a table (both sensors steady) and
   taken off, and rates under 0.007 turns a second count as none, as the console's standard zero-drift mode does (the
   Motion deadzone row: Low and High are its tight 0.002 and loose 0.01).
   Motion sensitivity X/Y and Invert motion X/Y scale the angular velocity before it is used: its part about the
   world's up (the turn, X) and the rest (the tilt, Y). At 100% and not inverted the state is Eden's own. Earlier
   versions sent a partial state (a yaw rate and a tilt matrix, the rest neutral, spluton's and Link-22's findings in
   issue 112), and every way the pad could be held at the start needed its own case; the full state needs none.

   The filter, the frame and the orientation matrix follow Eden's src/hid_core/frontend/motion_input.cpp and
   src/input_common/drivers/sdl_driver.cpp (Copyright 2020 yuzu Emulator Project, GPL-2.0-or-later; Copyright 2026 Eden
   Emulator Project, GPL-3.0-or-later), as this project is GPL-3.0. */
#pragma once
#include <math.h>
#include <stdint.h>
#include <string.h>

#define HOS_MOTION_G 9.80665f
#define HOS_MOTION_TAU 6.28318531f

typedef struct {
    float accel[3];       /* G, the Switch's frame */
    float gyro[3];        /* turns a second */
    float rotation[3];    /* turns */
    float direction[9];   /* rows: the orientation */
} HosMotionState;

/* one reading's state as GetSixAxisSensorStates hands it out: n counts the readings, dt_ns is the time since the one
   before (host/pad_sdl.c keeps the last HOS_MOTION_HISTORY) */
typedef struct {
    HosMotionState st;
    int64_t n, dt_ns;
} HosMotionSample;
#define HOS_MOTION_HISTORY 32

typedef struct {
    float quat[4];        /* x, y, z, w (Eden's order): the real orientation, Eden's filter on the real rates */
    float quat_out[4];    /* the orientation handed out: integrated from the rates as Motion sensitivity and Invert
                             motion scale them, with the real filter's drift correction (the same as quat at 100%) */
    float real_error[3], integral_error[3];
    float bias[3];        /* turns a second, the Switch's frame */
    float learned_t;      /* seconds of stillness learned from (fast at first, then slowly) */
    float still_t;        /* seconds still */
    float gmean[3], gvar, avec[3], avar, amean;   /* the short window's means and variances (stillness) */
    int stats_ready, only_accelerometer, reset_counter;
    float last_accel[3], last_gyro[3], pending_dt;
    float threshold;      /* turns a second, the deadzone (0: HOS_MOTION_THRESHOLD) */
    float up[3];          /* the world's up in the Switch's frame (from the accelerometer, smoothed: for the split of
                             the angular velocity into turn and tilt) */
    HosMotionState out;
} HosMotion;

#define HOS_MOTION_STILL_GYRO 0.015f    /* rad/s, summed over three axes (about 0.009 on each) */
#define HOS_MOTION_STILL_ACCEL 0.0173f  /* G, the same way (0.01 on each) */
#define HOS_MOTION_STILL_S 0.5f
#define HOS_MOTION_WINDOW_S 0.2f
#define HOS_MOTION_DT_MAX 0.1f          /* (Eden ignores a longer gap) */
#define HOS_MOTION_THRESHOLD 0.007f     /* turns a second: less is none (the console's standard zero-drift mode) */
#define HOS_MOTION_THRESHOLD_TIGHT 0.002f   /* (its tight and loose modes: the Motion deadzone row's Low and High) */
#define HOS_MOTION_THRESHOLD_LOOSE 0.01f

static inline void hos_motion_rest(HosMotionState *o) {
    memset(o, 0, sizeof *o);
    o->accel[2] = -1;
    o->direction[0] = o->direction[4] = o->direction[8] = 1;
}

static inline void hos_motion_reset(HosMotion *m) {
    memset(m, 0, sizeof *m);
    m->quat[2] = -1;   /* (Eden's ResetQuaternion: {0, 0, -1, 0}) */
    m->quat_out[2] = -1;
    m->only_accelerometer = 1;
    m->up[2] = 1;
    hos_motion_rest(&m->out);
}

static inline float hos_motion_len(const float v[3]) { return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }

/* a quaternion (Eden's x, y, z, w) turned at Eden's filter rate rg for period seconds */
static inline void hos_motion_integrate(float quat[4], const float rg[3], float period) {
    float q1 = quat[3], q2 = quat[0], q3 = quat[1], q4 = quat[2];
    const float gx = rg[1], gy = rg[0], gz = rg[2];
    const float pa = q2, pb = q3, pc = q4;
    q1 = q1 + (-q2 * gx - q3 * gy - q4 * gz) * (0.5f * period);
    q2 = pa + (q1 * gx + pb * gz - pc * gy) * (0.5f * period);
    q3 = pb + (q1 * gy - pa * gz + pc * gx) * (0.5f * period);
    q4 = pc + (q1 * gz + pa * gy - pb * gx) * (0.5f * period);
    const float n = sqrtf(q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4);
    if (n > 0) quat[3] = q1 / n, quat[0] = q2 / n, quat[1] = q3 / n, quat[2] = q4 / n;
}
static inline void hos_motion_rate(const HosMotion *m, const float gyro[3], float rg[3]) {   /* (Eden's swap, rad/s) */
    rg[0] = gyro[1] * HOS_MOTION_TAU, rg[1] = -gyro[0] * HOS_MOTION_TAU, rg[2] = -gyro[2] * HOS_MOTION_TAU;
    if (m->only_accelerometer) rg[0] = rg[1] = rg[2] = 0;
}

/* Mahony's filter on the real quaternion, Eden's UpdateOrientation (period: seconds; gain 10: the settling of
   SetOrientationFromAccelerometer). Returns the feedback it added to the rate, in *corr (rad/s, the filter's frame), so
   the handed-out orientation can lose the same drift. */
static inline void hos_motion_mahony(HosMotion *m, const float accel[3], const float gyro[3], float period, float gain,
                                     float corr[3]) {
    const float q1 = m->quat[3], q2 = m->quat[0], q3 = m->quat[1], q4 = m->quat[2];
    const float alen = hos_motion_len(accel);
    float rg[3];
    hos_motion_rate(m, gyro, rg);
    if (gain > 1) rg[0] = rg[1] = rg[2] = 0;
    corr[0] = corr[1] = corr[2] = 0;
    if ((alen >= 0.75f && alen <= 1.25f) || gain > 1) {
        const float ax = -accel[0] / alen, ay = accel[1] / alen, az = -accel[2] / alen;
        const float vx = 2.0f * (q2 * q4 - q1 * q3);
        const float vy = 2.0f * (q1 * q2 + q3 * q4);
        const float vz = q1 * q1 - q2 * q2 - q3 * q3 + q4 * q4;
        const float e[3] = {az * vx - ax * vz, ay * vz - az * vy, ax * vy - ay * vx};
        memcpy(m->real_error, e, sizeof e);
        if (gain <= 1 && hos_motion_len(e) >= 0.05f)
            for (int i = 0; i < 3; i++) m->integral_error[i] += e[i];
        else if (gain <= 1)
            memset(m->integral_error, 0, sizeof m->integral_error);
        const float kp = 0.3f, ki = 0.005f;
        const float wp = gain > 1 ? 10 : m->only_accelerometer ? 35 : 1, wi = gain > 1 ? 5 : m->only_accelerometer ? 10 : 1;
        for (int i = 0; i < 3; i++) corr[i] = wp * kp * e[i] + wi * ki * m->integral_error[i];
    }
    for (int i = 0; i < 3; i++) rg[i] += corr[i];
    hos_motion_integrate(m->quat, rg, period);
}

/* the orientation matrix, Eden's GetOrientation (its quaternion's matrix, with its sign changes) */
static inline void hos_motion_direction(const float quat[4], float d[9]) {
    const float x = -quat[1], y = -quat[0], z = -quat[3], w = -quat[2];
    const float x2 = x * x, y2 = y * y, z2 = z * z, xy = x * y, wz = w * z, xz = x * z, wy = w * y, yz = y * z, wx = w * x;
    const float mm[9] = {1 - 2 * (y2 + z2), 2 * (xy + wz), 2 * (xz - wy), 2 * (xy - wz), 1 - 2 * (x2 + z2), 2 * (yz + wx),
                         2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (x2 + y2)};
    d[0] = mm[0], d[1] = mm[1], d[2] = -mm[2];
    d[3] = mm[3], d[4] = mm[4], d[5] = -mm[5];
    d[6] = -mm[6], d[7] = -mm[7], d[8] = mm[8];
}

/* One reading: SDL's accelerometer (m/s^2) and gyroscope (rad/s), dt seconds after the last one. sens_h and sens_v
   scale the turn and the tilt (1: as the pad turns); inv_h and inv_v turn them the other way. Returns 0 for a repeat
   of the last reading (no new reading: its time goes into the next), 1 otherwise. */
static inline int hos_motion_update(HosMotion *m, const float accel_sdl[3], const float gyro_sdl[3], float dt,
                                    float sens_h, float sens_v, int inv_h, int inv_v) {
    if (!(dt >= 0)) dt = 0;
    if (m->stats_ready && !memcmp(accel_sdl, m->last_accel, sizeof m->last_accel) &&
        !memcmp(gyro_sdl, m->last_gyro, sizeof m->last_gyro)) {
        m->pending_dt += dt;
        return 0;
    }
    memcpy(m->last_accel, accel_sdl, sizeof m->last_accel), memcpy(m->last_gyro, gyro_sdl, sizeof m->last_gyro);
    dt += m->pending_dt, m->pending_dt = 0;
    /* the Switch's frame and units (Eden's sdl_driver.cpp), clamped as Eden clamps them */
    float a[3] = {-accel_sdl[0] / HOS_MOTION_G, accel_sdl[2] / HOS_MOTION_G, -accel_sdl[1] / HOS_MOTION_G};
    float g[3] = {gyro_sdl[0] / HOS_MOTION_TAU, -gyro_sdl[2] / HOS_MOTION_TAU, gyro_sdl[1] / HOS_MOTION_TAU};
    for (int i = 0; i < 3; i++) {
        a[i] = a[i] < -7 ? -7 : a[i] > 7 ? 7 : a[i];
    }
    const float alen = hos_motion_len(a);
    /* stillness: the short window's spread of both sensors (rad/s and G, summed over the axes), about 1 G */
    if (!m->stats_ready) {
        memcpy(m->gmean, g, sizeof m->gmean), memcpy(m->avec, a, sizeof m->avec);
        m->gvar = 0.01f, m->avar = 0.01f, m->amean = alen, m->stats_ready = 1;
    } else if (dt > 0) {
        const float k = dt / (HOS_MOTION_WINDOW_S + dt);
        float gv = 0, av = 0;
        for (int i = 0; i < 3; i++) {
            const float dg = (g[i] - m->gmean[i]) * HOS_MOTION_TAU, da = a[i] - m->avec[i];
            m->gmean[i] += (g[i] - m->gmean[i]) * k, m->avec[i] += da * k;
            gv += dg * dg, av += da * da;
        }
        m->gvar += (gv - m->gvar) * k, m->avar += (av - m->avar) * k, m->amean += (alen - m->amean) * k;
    }
    const int still = m->gvar < HOS_MOTION_STILL_GYRO * HOS_MOTION_STILL_GYRO &&
                      m->avar < HOS_MOTION_STILL_ACCEL * HOS_MOTION_STILL_ACCEL && m->amean > 0.9f && m->amean < 1.1f;
    m->still_t = still ? m->still_t + dt : 0;
    if (m->still_t >= HOS_MOTION_STILL_S && dt > 0) {
        const float k = dt / ((m->learned_t < 2 ? 0.25f : 3.0f) + dt);
        for (int i = 0; i < 3; i++) m->bias[i] += (m->gmean[i] - m->bias[i]) * k;
        m->learned_t += dt;
    }
    for (int i = 0; i < 3; i++) {
        g[i] -= m->bias[i];
        g[i] = g[i] < -5 ? -5 : g[i] > 5 ? 5 : g[i];
    }
    if (hos_motion_len(g) < (m->threshold > 0 ? m->threshold : HOS_MOTION_THRESHOLD)) g[0] = g[1] = g[2] = 0;
    else m->only_accelerometer = 0;
    /* the turn and the tilt: the angular velocity's part about the world's up, and the rest, scaled and turned around
       as the menu says (the up from the accelerometer, smoothed) */
    if (alen > 0.65f && alen < 1.35f) {
        const float k = dt / (0.1f + dt);
        for (int i = 0; i < 3; i++) m->up[i] += (-a[i] / alen - m->up[i]) * k;
        const float n = hos_motion_len(m->up);
        if (n > 0.001f) for (int i = 0; i < 3; i++) m->up[i] /= n;
    }
    float gs[3];   /* (the scaled rates: what the game is handed; g stays real for the filter) */
    {
        const float sh = sens_h * (inv_h ? -1.0f : 1.0f), sv = sens_v * (inv_v ? -1.0f : 1.0f);
        const float t = g[0] * m->up[0] + g[1] * m->up[1] + g[2] * m->up[2];
        for (int i = 0; i < 3; i++) gs[i] = sh * t * m->up[i] + sv * (g[i] - t * m->up[i]);
    }
    if (dt > HOS_MOTION_DT_MAX) dt = 0;   /* (Eden ignores a stall: no rotation, no orientation step) */
    for (int i = 0; i < 3; i++) m->out.rotation[i] += gs[i] * dt;
    /* Eden's ResetOrientation: a filter that lost track (its error over 0.1) is set again from the accelerometer once
       the pad has lain face up and still for 900 readings */
    if (hos_motion_len(m->real_error) >= 0.1f && !m->only_accelerometer) {
        if (hos_motion_len(g) < 0.05f && alen > 0.9f && alen < 1.1f && a[2] <= -0.9f) {
            if (++m->reset_counter > 900) {
                m->quat[0] = m->quat[1] = m->quat[3] = 0, m->quat[2] = -1;
                memset(m->integral_error, 0, sizeof m->integral_error);
                float c[3];
                for (int it = 0; it < 100 && hos_motion_len(m->real_error) >= 0.01f; it++)
                    hos_motion_mahony(m, a, g, 0.015f, 10, c);
                memcpy(m->quat_out, m->quat, sizeof m->quat);   /* (set again from gravity: the handed-out one too) */
                m->reset_counter = 0;
            }
        } else {
            m->reset_counter = 0;
        }
    }
    if (dt > 0) {
        /* the real filter, then the handed-out orientation: the scaled rates plus the real filter's correction, so it
           loses the gyro's drift but keeps a scaled or inverted move (the gravity pull acts on the real one only;
           hckrman101). While only the accelerometer has been seen the two settle together. */
        const int settling = m->only_accelerometer;
        float c[3], rs[3];
        hos_motion_mahony(m, a, g, dt, 1, c);
        if (settling) {
            memcpy(m->quat_out, m->quat, sizeof m->quat);
        } else {
            hos_motion_rate(m, gs, rs);
            for (int i = 0; i < 3; i++) rs[i] += c[i];
            hos_motion_integrate(m->quat_out, rs, dt);
        }
    }
    memcpy(m->out.accel, a, sizeof a);
    memcpy(m->out.gyro, gs, sizeof gs);
    hos_motion_direction(m->quat_out, m->out.direction);
    return 1;
}
