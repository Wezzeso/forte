# Shared PC control lease

Status recorded 17 October 2026, Asia/Qyzylorda. One operator may issue shared-device GUI input at a time.

```yaml
controller_agent: null
last_controller_agent: a17_pc_gpu_deployment
purpose: no_active_gui_input_provider_and_dokploy_work_complete
initial_observed_target: chrome_tab_535945604_https_brev_nvidia_com_environment_new_public_gpu_A100
target: null
lease_started_at: 2026-10-17T14:55:38+05:00
next_handoff: none_requested_a19_completed_without_gui_input
input_exclusivity: exclusive
observed_capability: supported_browser_use_chrome
native_control_used: false
lease_state: released_on_a17_completion_handoff
```

A19 uses public web fetch/search for brand research and does not need GUI input. Before native/browser input is transferred, A17 must stop actions, record the last observed state, and acknowledge the handoff. A17 owns authentication and provider capability reporting. This lease grants no spending authority; unresolved budgets remain unresolved in the infrastructure ledger.

A17 completed its current GUI work and released the lease. The project-specific Dokploy ingress is verified running with private request logging disabled; final public UI/assets/authentication checks pass while model readiness remains false. A future GUI operator must reacquire the lease and inspect a fresh observed target. Remaining model/offline checks use authorized SSH.

A17's initial inventory found no local NVIDIA GPU and prepared a Brev quote. That initial no-resource state is superseded by the verified launch handoff below. Quote and resource details belong to A17's ledger; opening the provider console alone does not establish GPU access.

The root handoff records the user's authorization for the prepared A100 deployment within a USD 50 total budget and project frontend deployment at `hackathon.wezeso.me` through the user's Dokploy server. A17 reacquired the lease for those tasks. Actual costs, deployment and judging keep-alive remain governed by A17's current resource ledger.

Current handoff: A100 80 GB PCIe hardware is verified (81,920 MiB; driver 565.57.01), with 384 GiB at USD 2.03/h. Recorded launch: 2026-10-17 15:01:25+05; authorized runtime ceiling: 20 h, ending 2026-10-18 11:01:25+05 (expected USD 40.60 within USD 50 cap). Model artifacts, CUDA/BF16, warm readiness and isolated synthetic-marker inference pass. The Dokploy route at `https://hackathon.wezeso.me` passes external TLS/UI/assets/auth preflight and readiness is 200. The first generated-MOV public job reached failed. The actual judging deadline and cleanup owner remain unresolved; successful public model-backed video completion is not yet established.

Only provider/public-brand pages and synthetic UI fixtures may appear in screenshots sent to cloud tools. Private videos, private annotations, original evidence, secrets, and unrelated windows are excluded. Use shell/SSH through execution tools rather than terminal UI automation.
