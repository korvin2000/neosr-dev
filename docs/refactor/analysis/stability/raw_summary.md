# Raw stability/NaN experiment outcomes (stopped early at session end; logs summarised by grep)

## NaN ablations (t4 agent, iteration 1; scratchpad/nan/*.log)
- abl_adamw_it1.log: NaN=0, last iter line: 
- abl_clamp_it1.log: NaN=0, last iter line: 
- abl_fp16_it1.log: NaN=0, last iter line: 
- abl_fp32_it1.log: NaN=0, last iter line: 
- abl_no_cons_it1.log: NaN=0, last iter line: 
- abl_no_ema_it1.log: NaN=0, last iter line: 
- abl_no_fdl_it1.log: NaN=0, last iter line: 
- abl_no_gan_it1.log: NaN=0, last iter line: 
- abl_no_ldl_it1.log: NaN=0, last iter line: 
- abl_no_mssim_it1.log: NaN=0, last iter line: 
- base_it50.log: NaN=0, last iter line: 
- dphase_fp16_it50.log: NaN=0, last iter line: 
- fixmssim_it1.log: NaN=0, last iter line: 
- fixmssim_it50.log: NaN=0, last iter line: 
- fp16fix_it50.log: NaN=0, last iter line: 
- fp16_it50.log: NaN=0, last iter line: 

## Stability experiment runs (E1 competitor own recipe, E2 competitor neosr-like hyperparams, E3 neosr current)
- stab_e3fix_s1: NaN/nan=0, last: iter:    300, d_out: out_d_real: 2.2656e+00 
- stab_e3fix_s2: NaN/nan=0, last: iter:     10, d_out: out_d_real: 1.5300e+02 
- stab_e3fix_s3: NaN/nan=1, last: , d_out: 
- stab_e3_s1: NaN/nan=1, last: , d_out: 
- stab_e3_s2: NaN/nan=1, last: , d_out: 
- stab_e3_s3: NaN/nan=1, last: , d_out: 
- stab_e3_s4: NaN/nan=1, last: , d_out: 
- stab_e3_s5: NaN/nan=0, last: iter:    300, d_out: out_d_real: 4.7070e-01 
- stab_e3_s6: NaN/nan=0, last: , d_out: 
- stab_e1_s1: NaN/nan=0, last: iter:     300,, d_out: 
- stab_e1_s2: NaN/nan=0, last: iter:     300,, d_out: 
- stab_e2_s1: NaN/nan=0, last: iter:     300,, d_out: d_real: 6.9644e-01
- stab_e2_s2: NaN/nan=0, last: iter:     300,, d_out: d_real: 6.9637e-01

Conclusion (Opus, from the rows above): with the CURRENT neosr code (E3) 4 of 6 seeds abort with NaN at start, while the
competitor survives 300 iterations both with its own recipe (E1) and with neosr-like aggressive hyperparameters (E2,
D logits ~0.7). => neosr instability is mechanical (MS-SSIM NaN, dead G-side GAN gradient, D eval-mode spectral norm,
accumulation), not primarily hyperparameters. stab_e3fix_* (partial MS-SSIM fix only) still showed 1 NaN of 3 and D logits up
to 153 at iter 10 -> the GAN fixes (G1.1/G1.2) are needed too. Full T6 playbook was NOT produced (workflow stopped at session end).
