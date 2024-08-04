# Distributed Continual Pretraining Troubleshooting Guide

## My miner/validator is not running, please help.

Follow these steps to troubleshoot your miner or validator:

1. Check to see if it is posting to its weight repo consistently every 10-30 minutes.
2. For validators: Are you setting weights?
3. Is your commune node responsive? Try using comx and compare its performance. If there is a network overload, you may need to run a local node (especially for read operations).
4. Are there any suspicious logs?
5. If no obvious solution shows, please post an issue on the GitHub repo or reach out to the team on the commune discord SN16.

### Your vtrust sucks.

We understand. It is a compromise between computational cost to run a validator node and vtrust. We are planning some future edits to the training/validation workflow that should see both computational cost go down and vtrust go up.


### Lock registrations please.

Is the subnet on fire? Yes? Still not locking regs.

### I have an issue that needs to be solved NOW NOW NOW

No. We work as fast as possible. Faster than that and we destabilize the subnet which affects total subnet incentive which will affect you as well as a miner/validator

### But it's urgent.

No, the team needs to plan and schedule code changes to ensure both adherence to long-term vision as well as not introducing un-tested hotfixes that can lead to more issues in the future.

### Your code breaks too much WTF.

We're solving a hard problem that has no plug and play solution. We have a balance of internal testing and experimental release tolerance. Too much caution and we have endless development cycles. Too quick releasing and we release half-baked things.
