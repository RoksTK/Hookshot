/* host/hos_motion.h with synthetic SDL readings: no SDL, controller or game. tests/test_motion.py builds and runs it.

   A simulated pad: its true up vector (the world's up in its frame) turns with the true rotation, the accelerometer
   reads it (times G), and the gyroscope reads the true rotation plus an offset (its bias) and noise (a tiny one, a
   real sensor's, so no two readings repeat; or a hand's tremor). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "hos_motion.h"

static int failures;
#define NEAR(a, b, tol) do { const double _a = (a), _b = (b); if (fabs(_a - _b) > (tol)) { \
    fprintf(stderr, "FAIL %s:%d: %s = %.5f, want %.5f\n", __FILE__, __LINE__, #a, _a, _b); failures++; } } while (0)

static float su[3];          /* the true up, in the pad's frame */
static float offset[3];      /* the gyroscope's bias */
static float tremor;         /* the noise's size, rad/s (each axis, uniform) */
static float anoise;         /* the accelerometer's, G (each axis, uniform; 0: a real sensor's least) */
static unsigned rng = 1;
static float noise(void) {
    rng = rng * 1664525u + 1013904223u;
    return ((float)(rng >> 8) / 16777216.0f * 2 - 1);
}

static void pose(float x, float y, float z) {
    const float n = sqrtf(x * x + y * y + z * z);
    su[0] = x / n, su[1] = y / n, su[2] = z / n;
}
#define FLAT 0, 1, 0        /* face up on a table: gravity's reaction on SDL's +Y */
#define UPRIGHT 0, 0, -1    /* held upright, face to the player: the top (SDL's -Z) up */

/* the true rotation (gx, gy, gz) rad/s for seconds, read every dt */
static void run_dt(HosMotion *m, float gx, float gy, float gz, float seconds, float dt, float sens_h, float sens_v,
                   int inv_h, int inv_v) {
    const int n = (int)lroundf(seconds / dt);
    for (int i = 0; i < n; i++) {
        const float w[3] = {gx, gy, gz};
        const float an = (anoise > 0 ? anoise * HOS_MOTION_G : 1e-5f);
        const float a[3] = {su[0] * HOS_MOTION_G + an * noise(), su[1] * HOS_MOTION_G + an * noise(),
                            su[2] * HOS_MOTION_G + an * noise()};
        float g[3];
        for (int k = 0; k < 3; k++) g[k] = w[k] + offset[k] + (tremor > 0 ? tremor : 1e-6f) * noise();
        hos_motion_update(m, a, g, dt, sens_h, sens_v, inv_h, inv_v);
        /* the truth turns: a world vector seen from a frame turning at w turns the other way */
        for (int s = 0; s < 10; s++) {
            const float h = dt / 10;
            const float p[3] = {su[0] + (-w[1] * su[2] + w[2] * su[1]) * h, su[1] + (-w[2] * su[0] + w[0] * su[2]) * h,
                                su[2] + (-w[0] * su[1] + w[1] * su[0]) * h};
            pose(p[0], p[1], p[2]);
        }
    }
}
static void run(HosMotion *m, float gx, float gy, float gz, float seconds, float sens_h, float sens_v, int inv_h,
                int inv_v) {
    run_dt(m, gx, gy, gz, seconds, 0.004f, sens_h, sens_v, inv_h, inv_v);
}
static void start(HosMotion *m, float x, float y, float z) {   /* a fresh filter, the pad at rest, no bias or tremor */
    hos_motion_reset(m);
    pose(x, y, z);
    offset[0] = offset[1] = offset[2] = 0;
    tremor = 0, anoise = 0;
}

static float dot3(const float a[3], const float b[3]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
static void same_direction(const HosMotion *a, const HosMotion *b, double tol) {
    for (int i = 0; i < 9; i++) NEAR(a->out.direction[i], b->out.direction[i], tol);
}

int main(void) {
    HosMotion m, n;

    /* before any reading: a pad lying still, face up */
    hos_motion_reset(&m);
    NEAR(m.out.accel[2], -1, 1e-6);
    NEAR(m.out.direction[0] + m.out.direction[4] + m.out.direction[8], 3, 1e-6);

    /* lying face up: (0, 0, -1) G in the Switch's frame, no rate, the direction the identity */
    start(&m, FLAT);
    run(&m, 0, 0, 0, 1, 1, 1, 0, 0);
    NEAR(m.out.accel[0], 0, 1e-3);
    NEAR(m.out.accel[1], 0, 1e-3);
    NEAR(m.out.accel[2], -1, 1e-3);
    NEAR(m.out.gyro[0] + m.out.gyro[1] + m.out.gyro[2], 0, 1e-6);
    NEAR(m.out.direction[0] + m.out.direction[4] + m.out.direction[8], 3, 0.01);

    /* the Switch's frame and units (Eden's sdl_driver.cpp): a turn about SDL's +Y (the pad flat, turning left) at one
       turn a second is +1 in z; about SDL's +X (the front up), +1 in x; held upright, SDL's -Z up, -Z is +y */
    start(&m, FLAT);
    run(&m, 0, 0, 0, 0.2f, 1, 1, 0, 0);
    run(&m, 0, HOS_MOTION_TAU, 0, 0.05f, 1, 1, 0, 0);
    NEAR(m.out.gyro[2], 1, 1e-3);
    NEAR(m.out.gyro[0], 0, 1e-3);
    start(&m, FLAT);
    run(&m, 0, 0, 0, 0.2f, 1, 1, 0, 0);
    run(&m, HOS_MOTION_TAU, 0, 0, 0.02f, 1, 1, 0, 0);
    NEAR(m.out.gyro[0], 1, 0.02);
    start(&m, UPRIGHT);
    run(&m, 0, 0, 0, 0.2f, 1, 1, 0, 0);
    NEAR(m.out.accel[1], -1, 1e-3);
    run(&m, 0, 0, -HOS_MOTION_TAU, 0.05f, 1, 1, 0, 0);
    NEAR(m.out.gyro[1], 1, 1e-3);

    /* the rotation is the angular velocity summed: half a second at a quarter turn a second about +z is 0.125 */
    start(&m, FLAT);
    run(&m, 0, 0, 0, 0.2f, 1, 1, 0, 0);
    run(&m, 0, HOS_MOTION_TAU / 4, 0, 0.5f, 1, 1, 0, 0);
    NEAR(m.out.rotation[2], 0.125, 3e-3);

    /* how the pad is held when the filter starts does not matter: started flat and then tilted 0.7 rad nose up, or
       started there, the direction is the same (Link-22's start-pose case: the console's orientation follows gravity) */
    start(&m, FLAT);
    run(&m, 0, 0, 0, 0.5f, 1, 1, 0, 0);
    run(&m, 1, 0, 0, 0.7f, 1, 1, 0, 0);
    run(&m, 0, 0, 0, 2, 1, 1, 0, 0);
    start(&n, su[0], su[1], su[2]);
    run(&n, 0, 0, 0, 2.5f, 1, 1, 0, 0);
    same_direction(&m, &n, 0.03);
    /* ... rolled 40 degrees and pitched 0.6 at the start, then levelled: the same as a level start in that pose */
    start(&m, 0, cosf(0.6f), -sinf(0.6f));
    {
        const float c = cosf(0.7f), sn = sinf(0.7f), x = su[0] * c - su[1] * sn, y = su[0] * sn + su[1] * c;
        pose(x, y, su[2]);
    }
    run(&m, 0, 0, 0, 0.5f, 1, 1, 0, 0);
    for (int i = 0; i < 80 && fabsf(su[0]) > 0.01f; i++) run(&m, 0, 0, su[0] > 0 ? -0.5236f : 0.5236f, 0.02f, 1, 1, 0, 0);
    run(&m, 0, 0, 0, 2, 1, 1, 0, 0);
    start(&n, su[0], su[1], su[2]);
    run(&n, 0, 0, 0, 2.5f, 1, 1, 0, 0);
    /* (the tilt agrees: the gravity each matrix implies; the heading about the vertical is the gyro's own, as on the
       console, so compare the matrices' third columns, the world's vertical seen from the pad) */
    {
        const float a3[3] = {m.out.direction[2], m.out.direction[5], m.out.direction[8]};
        const float b3[3] = {n.out.direction[2], n.out.direction[5], n.out.direction[8]};
        NEAR(dot3(a3, b3), 1, 2e-3);
    }

    /* the gyroscope's resting rate is learned on a table and taken off */
    start(&m, FLAT);
    offset[0] = 0.012f, offset[1] = -0.008f, offset[2] = 0.010f;
    run(&m, 0, 0, 0, 4, 1, 1, 0, 0);
    NEAR(m.bias[0], 0.012 / HOS_MOTION_TAU, 2e-4);
    NEAR(m.bias[2], -0.008 / HOS_MOTION_TAU, 2e-4);   /* (SDL's y is the Switch's z) */
    NEAR(m.out.gyro[0] + m.out.gyro[1] + m.out.gyro[2], 0, 1e-6);
    /* ... a large one too, and a noisier sensor's (0.0067 rad/s and 5 mG on each axis) */
    start(&m, FLAT);
    offset[1] = 0.2f;
    run(&m, 0, 0, 0, 4, 1, 1, 0, 0);
    NEAR(m.bias[2], 0.2 / HOS_MOTION_TAU, 2e-4);
    start(&m, FLAT);
    offset[1] = 0.02f, tremor = 0.0116f, anoise = 0.0087f;
    run(&m, 0, 0, 0, 6, 1, 1, 0, 0);
    NEAR(m.bias[2], 0.02 / HOS_MOTION_TAU, 4e-4);
    /* ... but a slow turn in the hand (tremor) is aim, and a steady machine pitch is not stillness: neither is learned */
    start(&m, FLAT);
    run(&m, 0, 0, 0, 1, 1, 1, 0, 0);
    tremor = 0.03f;
    run(&m, 0, 0.0524f, 0, 2, 1, 1, 0, 0);
    NEAR(m.bias[2], 0, 2e-4);
    start(&m, FLAT);
    run(&m, 0.2f, 0, 0, 3, 1, 1, 0, 0);
    NEAR(m.bias[0], 0, 2e-4);

    /* under the console's zero-drift threshold (0.007 turns a second) a rate counts as none */
    start(&m, FLAT);
    run(&m, 0, 0, 0, 0.2f, 1, 1, 0, 0);
    run(&m, 0, 0.03f, 0, 0.1f, 1, 1, 0, 0);
    NEAR(m.out.gyro[2], 0, 1e-6);

    /* the Motion deadzone row: Low (the console's tight mode) lets a 0.005 turns/s turn through, Standard does not,
       High (loose) also drops 0.008 */
    start(&m, FLAT);
    run(&m, 0, 0, 0, 0.2f, 1, 1, 0, 0);
    m.threshold = HOS_MOTION_THRESHOLD_TIGHT;
    run(&m, 0, 0.03f, 0, 0.1f, 1, 1, 0, 0);
    NEAR(m.out.gyro[2], 0.03 / HOS_MOTION_TAU, 1e-4);
    m.threshold = HOS_MOTION_THRESHOLD_LOOSE;
    run(&m, 0, 0.05f, 0, 0.1f, 1, 1, 0, 0);
    NEAR(m.out.gyro[2], 0, 1e-6);
    m.threshold = 0;

    /* the sensitivities and the inverts: the turn (about the world's up) and the tilt (the rest) each scaled */
    start(&m, FLAT);
    run(&m, 0, 0, 0, 0.3f, 1, 1, 0, 0);
    run(&m, 0, HOS_MOTION_TAU, 0, 0.05f, 2, 0.5f, 0, 0);
    NEAR(m.out.gyro[2], 2, 0.01);
    run(&m, 0, HOS_MOTION_TAU, 0, 0.05f, 1, 1, 1, 0);
    NEAR(m.out.gyro[2], -1, 0.01);
    start(&m, FLAT);
    run(&m, 0, 0, 0, 0.3f, 1, 1, 0, 0);
    run(&m, HOS_MOTION_TAU, 0, 0, 0.01f, 3, 0.5f, 0, 0);
    NEAR(m.out.gyro[0], 0.5, 0.02);
    NEAR(m.out.gyro[2], 0, 0.01);
    run(&m, HOS_MOTION_TAU, 0, 0, 0.01f, 1, 1, 0, 1);
    NEAR(m.out.gyro[0], -1, 0.03);

    /* ... and the orientation keeps a scaled or inverted tilt while held: 0.5 rad nose up at 200% stays at 1.0 for 3 s,
       and inverted at -0.5 (the gravity pull acts on the real filter only; hckrman101). Compared against the matrix of
       a pad really tilted that far, at 100% */
    for (int c = 0; c < 2; c++) {
        const float sens = c ? 1.0f : 2.0f, want = c ? -0.5f : 1.0f;
        start(&m, FLAT);
        run(&m, 0, 0, 0, 0.5f, 1, sens, 0, c);
        run(&m, 1, 0, 0, 0.5f, 1, sens, 0, c);
        run(&m, 0, 0, 0, 3, 1, sens, 0, c);
        start(&n, FLAT);
        run(&n, 0, 0, 0, 0.5f, 1, 1, 0, 0);
        run(&n, want > 0 ? 1.0f : -1.0f, 0, 0, fabsf(want), 1, 1, 0, 0);
        run(&n, 0, 0, 0, 3, 1, 1, 0, 0);
        same_direction(&m, &n, 0.04);
    }

    /* a repeat of the last reading is no reading (its time goes into the next), and a stall is no step */
    start(&m, FLAT);
    run(&m, 0, 0, 0, 0.2f, 1, 1, 0, 0);
    {
        const float a[3] = {0, HOS_MOTION_G, 0}, g0[3] = {0, 0, 0}, g1[3] = {0, HOS_MOTION_TAU, 0};
        NEAR(hos_motion_update(&m, a, g0, 0.004f, 1, 1, 0, 0), 1, 0);
        NEAR(hos_motion_update(&m, a, g0, 0.004f, 1, 1, 0, 0), 0, 0);
        const float r0 = m.out.rotation[2];
        NEAR(hos_motion_update(&m, a, g1, 0.004f, 1, 1, 0, 0), 1, 0);
        NEAR(m.out.rotation[2] - r0, 0.008, 2e-4);
        const float r1 = m.out.rotation[2];
        hos_motion_update(&m, a, g0, 1.0f, 1, 1, 0, 0);
        NEAR(m.out.rotation[2], r1, 1e-6);
    }

    if (failures) return 1;
    puts("motion: ok");
    return 0;
}
