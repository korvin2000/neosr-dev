// Stand-in for `train.py` in trainer-runner.spec.ts: writes trainer-format lines to stderr.
// Usage: node fake-train.mjs <replay|fail|startup|teardown|loop|endonint|stubborn> <toml> [--auto_resume]
// loop and endonint start at the update of the TOML's `resume_state` (`<update>.state`), as train.py does.
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const [mode, toml] = process.argv.slice(2);
const ts = '11-11-2026 09:00 AM |';
const say = (line) => process.stderr.write(`${line}\n`);
process.stderr.on('error', () => {}); // like Python's logging: a closed pipe (dead backend) does not stop training
say(`${ts} INFO: fake trainer for ${path.basename(toml)}`);

if (mode === 'replay') {
  process.stderr.write(fs.readFileSync(path.join(import.meta.dirname, 'p0_pretrain.log'), 'utf8'));
} else if (mode === 'fail') {
  say(`${ts} INFO: Starting at epoch 0, update 0.`);
  say(`${ts} ERROR: training failed`);
  say('Traceback (most recent call last):');
  say('  File "train.py", line 1, in main');
  say('FloatingPointError: non-finite weights or buffers; the last committed checkpoint stands');
  process.exitCode = 1;
} else if (mode === 'teardown') {
  // finished, then a slow interpreter exit: a SIGINT now kills it
  say(`${ts} INFO: Starting at epoch 0, update 0.`);
  say(`${ts} INFO: End of training at update 100. Time consumed: 0:00:01`);
  setInterval(() => {}, 1000);
} else if (mode === 'startup') {
  setInterval(() => {}, 1000); // "importing torch": no SIGINT handler yet, so the signal kills it
} else {
  // loop: progress every 20 ms, SIGINT stops gracefully; endonint: SIGINT during the last update
  // (the run ends normally); stubborn: ignores SIGINT and has a child process.
  const resumed = /resume_state = ".*?(\d+)\.state"/.exec(fs.readFileSync(toml, 'utf8'));
  let update = resumed ? Number(resumed[1]) : 0;
  say(`${ts} INFO: Starting at epoch 0, update ${update}.`);
  const timer = setInterval(() => {
    update += 100;
    say(`${ts} INFO: [ epoch:   0 ] [ iter: ${update.toLocaleString('en-US').padStart(7)} ] [ performance: 10.000 it/s, data 0% ] [ lr: 2.00e-04 ] [ eta: 0:01:00 ] [ l_g_total: 1.0000e-01 ] `);
  }, 20);
  if (mode === 'stubborn') {
    const child = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { stdio: 'ignore' });
    process.stdout.write(`grandchild ${child.pid}\n`);
    process.on('SIGINT', () => {});
  } else {
    process.on('SIGINT', () => {
      clearInterval(timer);
      if (mode === 'endonint') {
        say(`${ts} INFO: End of training at update ${update}. Time consumed: 0:00:01`);
        return;
      }
      say(`${ts} INFO: stop requested: saving update ${update}`);
      say(`${ts} INFO: Stopped at update ${update}. Time consumed: 0:00:01`);
    });
  }
}
