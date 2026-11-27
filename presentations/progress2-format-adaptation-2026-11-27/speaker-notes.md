# Progress 2 speaker notes

## Slide 1

This update reports a completed, independently audited fixed-format adaptation comparison. The target is the fictional sender's attribution of the reader's current location belief. It is not a measurement of spontaneous anticipation or deception. Human semantic review remains incomplete.

Reporting inputs (no new inference or analysis):
{
  "summary": {
    "path": "C:\\Users\\adirj\\OneDrive\\Documents\\GitHub\\6CCSAPRJ - Individual project\\Individual-Project\\docs\\results\\format-adaptation-2026-11-27\\summary.json",
    "sha256": "b20171e7c77ad704b37c29bec1aabe6bfcb5d7d60a38f4c5b9fe6132dc81392e"
  },
  "manifest": {
    "path": "C:\\Users\\adirj\\OneDrive\\Documents\\GitHub\\6CCSAPRJ - Individual project\\Individual-Project\\docs\\results\\format-adaptation-2026-11-27\\manifest.json",
    "sha256": "3d03d215a3f8e56abbe1aa76f708a072992caff4c33088f8bff32e90b3fce67a"
  },
  "audit": {
    "path": "C:\\Users\\adirj\\OneDrive\\Documents\\GitHub\\6CCSAPRJ - Individual project\\Individual-Project\\docs\\results\\format-adaptation-2026-11-27\\audit_receipt.json",
    "sha256": "1daa87e5692a6e04aacd46750b5919ba80e08dca94de7b2f3dcc492227fe660f"
  },
  "campaign_identity": {
    "path": "/home/adirj/kcl-latent-probes-runs/format_adaptation_v1_20261127_campaign",
    "sha256": null
  }
}

## Slide 2

All counts use complete B1-by-B2 receipt quartets, not independent row accuracy. The primary is Llama Mixed512 minus A512 in B hard correction; Qwen is a named secondary replication. The 95% percentile intervals resample the same 24 paired contexts 10,000 times at seed 20261127. They exclude uncertainty from resampling the fitting cohort. A512 and Mixed512 use the same 512 formal states; Mixed uses 256 A plus 256 B observations. Ahalf uses 256 A; Offset freezes those coefficients and fits only B intercepts. Mixed versus Offset compares these procedures and cannot uniquely identify geometric rotation or internal mechanism. Both format controls and every fixed arm remain visible regardless of the effect's sign.

Saved contrasts:
{
  "llama": {
    "primary": {
      "contrast": "Mixed512_minus_A512",
      "renderer": "B",
      "condition": "correction_hard",
      "endpoint": "complete_four_case_quartet_accuracy",
      "groups": 24,
      "quartets": 192,
      "Mixed512": 0.7864583333333334,
      "A512": 0.010416666666666666,
      "difference": 0.7760416666666666,
      "conditional_percentile_interval_95": [
        0.6822916666666666,
        0.8541666666666666
      ],
      "bootstrap_samples": 10000,
      "bootstrap_seed": 20261127,
      "uncertainty_scope": "Paired whole contexts conditional on frozen authored bank, old feature map, site and fitted heads; excludes coefficient/checkpoint uncertainty",
      "group_values": [
        {
          "group_id": "adapt_test_00",
          "Mixed512": 0.75,
          "A512": 0.0,
          "difference": 0.75
        },
        {
          "group_id": "adapt_test_01",
          "Mixed512": 0.75,
          "A512": 0.0,
          "difference": 0.75
        },
        {
          "group_id": "adapt_test_02",
          "Mixed512": 1.0,
          "A512": 0.0,
          "difference": 1.0
        },
        {
          "group_id": "adapt_test_03",
          "Mixed512": 0.625,
          "A512": 0.0,
          "difference": 0.625
        },
        {
          "group_id": "adapt_test_04",
          "Mixed512": 0.625,
          "A512": 0.0,
          "difference": 0.625
        },
        {
          "group_id": "adapt_test_05",
          "Mixed512": 0.625,
          "A512": 0.0,
          "difference": 0.625
        },
        {
          "group_id": "adapt_test_06",
          "Mixed512": 1.0,
          "A512": 0.0,
          "difference": 1.0
        },
        {
          "group_id": "adapt_test_07",
          "Mixed512": 1.0,
          "A512": 0.0,
          "difference": 1.0
        },
        {
          "group_id": "adapt_test_08",
          "Mixed512": 0.75,
          "A512": 0.0,
          "difference": 0.75
        },
        {
          "group_id": "adapt_test_09",
          "Mixed512": 0.625,
          "A512": 0.0,
          "difference": 0.625
        },
        {
          "group_id": "adapt_test_10",
          "Mixed512": 0.875,
          "A512": 0.0,
          "difference": 0.875
        },
        {
          "group_id": "adapt_test_11",
          "Mixed512": 0.75,
          "A512": 0.0,
          "difference": 0.75
        },
        {
          "group_id": "adapt_test_12",
          "Mixed512": 0.625,
          "A512": 0.0,
          "difference": 0.625
        },
        {
          "group_id": "adapt_test_13",
          "Mixed512": 1.0,
          "A512": 0.0,
          "difference": 1.0
        },
        {
          "group_id": "adapt_test_14",
          "Mixed512": 0.75,
          "A512": 0.125,
          "difference": 0.625
        },
        {
          "group_id": "adapt_test_15",
          "Mixed512": 0.875,
          "A512": 0.0,
          "difference": 0.875
        },
        {
          "group_id": "adapt_test_16",
          "Mixed512": 1.0,
          "A512": 0.125,
          "difference": 0.875
        },
        {
          "group_id": "adapt_test_17",
          "Mixed512": 0.875,
          "A512": 0.0,
          "difference": 0.875
        },
        {
          "group_id": "adapt_test_18",
          "Mixed512": 0.875,
          "A512": 0.0,
          "difference": 0.875
        },
        {
          "group_id": "adapt_test_19",
          "Mixed512": 0.0,
          "A512": 0.0,
          "difference": 0.0
        },
        {
          "group_id": "adapt_test_20",
          "Mixed512": 0.75,
          "A512": 0.0,
          "difference": 0.75
        },
        {
          "group_id": "adapt_test_21",
          "Mixed512": 0.75,
          "A512": 0.0,
          "difference": 0.75
        },
        {
          "group_id": "adapt_test_22",
          "Mixed512": 1.0,
          "A512": 0.0,
          "difference": 1.0
        },
        {
          "group_id": "adapt_test_23",
          "Mixed512": 1.0,
          "A512": 0.0,
          "difference": 1.0
        }
      ],
      "role": "primary"
    },
    "named_diagnostic": {
      "contrast": "Mixed512_minus_Offset",
      "renderer": "B",
      "condition": "correction_hard",
      "endpoint": "complete_four_case_quartet_accuracy",
      "groups": 24,
      "quartets": 192,
      "Mixed512": 0.7864583333333334,
      "Offset": 0.10416666666666667,
      "difference": 0.6822916666666666,
      "conditional_percentile_interval_95": [
        0.5885416666666666,
        0.7604166666666666
      ],
      "bootstrap_samples": 10000,
      "bootstrap_seed": 20261127,
      "uncertainty_scope": "Paired whole contexts conditional on frozen authored bank, old feature map, site and fitted heads; excludes coefficient/checkpoint uncertainty",
      "group_values": [
        {
          "group_id": "adapt_test_00",
          "Mixed512": 0.75,
          "Offset": 0.0,
          "difference": 0.75
        },
        {
          "group_id": "adapt_test_01",
          "Mixed512": 0.75,
          "Offset": 0.0,
          "difference": 0.75
        },
        {
          "group_id": "adapt_test_02",
          "Mixed512": 1.0,
          "Offset": 0.125,
          "difference": 0.875
        },
        {
          "group_id": "adapt_test_03",
          "Mixed512": 0.625,
          "Offset": 0.0,
          "difference": 0.625
        },
        {
          "group_id": "adapt_test_04",
          "Mixed512": 0.625,
          "Offset": 0.0,
          "difference": 0.625
        },
        {
          "group_id": "adapt_test_05",
          "Mixed512": 0.625,
          "Offset": 0.25,
          "difference": 0.375
        },
        {
          "group_id": "adapt_test_06",
          "Mixed512": 1.0,
          "Offset": 0.0,
          "difference": 1.0
        },
        {
          "group_id": "adapt_test_07",
          "Mixed512": 1.0,
          "Offset": 0.0,
          "difference": 1.0
        },
        {
          "group_id": "adapt_test_08",
          "Mixed512": 0.75,
          "Offset": 0.0,
          "difference": 0.75
        },
        {
          "group_id": "adapt_test_09",
          "Mixed512": 0.625,
          "Offset": 0.125,
          "difference": 0.5
        },
        {
          "group_id": "adapt_test_10",
          "Mixed512": 0.875,
          "Offset": 0.125,
          "difference": 0.75
        },
        {
          "group_id": "adapt_test_11",
          "Mixed512": 0.75,
          "Offset": 0.25,
          "difference": 0.5
        },
        {
          "group_id": "adapt_test_12",
          "Mixed512": 0.625,
          "Offset": 0.0,
          "difference": 0.625
        },
        {
          "group_id": "adapt_test_13",
          "Mixed512": 1.0,
          "Offset": 0.25,
          "difference": 0.75
        },
        {
          "group_id": "adapt_test_14",
          "Mixed512": 0.75,
          "Offset": 0.0,
          "difference": 0.75
        },
        {
          "group_id": "adapt_test_15",
          "Mixed512": 0.875,
          "Offset": 0.125,
          "difference": 0.75
        },
        {
          "group_id": "adapt_test_16",
          "Mixed512": 1.0,
          "Offset": 0.25,
          "difference": 0.75
        },
        {
          "group_id": "adapt_test_17",
          "Mixed512": 0.875,
          "Offset": 0.375,
          "difference": 0.5
        },
        {
          "group_id": "adapt_test_18",
          "Mixed512": 0.875,
          "Offset": 0.25,
          "difference": 0.625
        },
        {
          "group_id": "adapt_test_19",
          "Mixed512": 0.0,
          "Offset": 0.0,
          "difference": 0.0
        },
        {
          "group_id": "adapt_test_20",
          "Mixed512": 0.75,
          "Offset": 0.125,
          "difference": 0.625
        },
        {
          "group_id": "adapt_test_21",
          "Mixed512": 0.75,
          "Offset": 0.125,
          "difference": 0.625
        },
        {
          "group_id": "adapt_test_22",
          "Mixed512": 1.0,
          "Offset": 0.125,
          "difference": 0.875
        },
        {
          "group_id": "adapt_test_23",
          "Mixed512": 1.0,
          "Offset": 0.0,
          "difference": 1.0
        }
      ],
      "interpretation": "Difference between fixed fitting procedures at equal observed A/B budget; not a unique geometric or mechanistic attribution"
    }
  },
  "qwen": {
    "primary": {
      "contrast": "Mixed512_minus_A512",
      "renderer": "B",
      "condition": "correction_hard",
      "endpoint": "complete_four_case_quartet_accuracy",
      "groups": 24,
      "quartets": 192,
      "Mixed512": 0.6510416666666666,
      "A512": 0.015625,
      "difference": 0.6354166666666666,
      "conditional_percentile_interval_95": [
        0.546875,
        0.7135416666666666
      ],
      "bootstrap_samples": 10000,
      "bootstrap_seed": 20261127,
      "uncertainty_scope": "Paired whole contexts conditional on frozen authored bank, old feature map, site and fitted heads; excludes coefficient/checkpoint uncertainty",
      "group_values": [
        {
          "group_id": "adapt_test_00",
          "Mixed512": 0.875,
          "A512": 0.0,
          "difference": 0.875
        },
        {
          "group_id": "adapt_test_01",
          "Mixed512": 0.75,
          "A512": 0.0,
          "difference": 0.75
        },
        {
          "group_id": "adapt_test_02",
          "Mixed512": 0.75,
          "A512": 0.0,
          "difference": 0.75
        },
        {
          "group_id": "adapt_test_03",
          "Mixed512": 0.75,
          "A512": 0.0,
          "difference": 0.75
        },
        {
          "group_id": "adapt_test_04",
          "Mixed512": 0.375,
          "A512": 0.0,
          "difference": 0.375
        },
        {
          "group_id": "adapt_test_05",
          "Mixed512": 0.875,
          "A512": 0.0,
          "difference": 0.875
        },
        {
          "group_id": "adapt_test_06",
          "Mixed512": 0.625,
          "A512": 0.0,
          "difference": 0.625
        },
        {
          "group_id": "adapt_test_07",
          "Mixed512": 0.5,
          "A512": 0.0,
          "difference": 0.5
        },
        {
          "group_id": "adapt_test_08",
          "Mixed512": 0.875,
          "A512": 0.0,
          "difference": 0.875
        },
        {
          "group_id": "adapt_test_09",
          "Mixed512": 0.75,
          "A512": 0.125,
          "difference": 0.625
        },
        {
          "group_id": "adapt_test_10",
          "Mixed512": 0.875,
          "A512": 0.0,
          "difference": 0.875
        },
        {
          "group_id": "adapt_test_11",
          "Mixed512": 0.125,
          "A512": 0.125,
          "difference": 0.0
        },
        {
          "group_id": "adapt_test_12",
          "Mixed512": 0.625,
          "A512": 0.0,
          "difference": 0.625
        },
        {
          "group_id": "adapt_test_13",
          "Mixed512": 0.625,
          "A512": 0.0,
          "difference": 0.625
        },
        {
          "group_id": "adapt_test_14",
          "Mixed512": 0.75,
          "A512": 0.125,
          "difference": 0.625
        },
        {
          "group_id": "adapt_test_15",
          "Mixed512": 0.625,
          "A512": 0.0,
          "difference": 0.625
        },
        {
          "group_id": "adapt_test_16",
          "Mixed512": 0.875,
          "A512": 0.0,
          "difference": 0.875
        },
        {
          "group_id": "adapt_test_17",
          "Mixed512": 0.75,
          "A512": 0.0,
          "difference": 0.75
        },
        {
          "group_id": "adapt_test_18",
          "Mixed512": 0.5,
          "A512": 0.0,
          "difference": 0.5
        },
        {
          "group_id": "adapt_test_19",
          "Mixed512": 0.25,
          "A512": 0.0,
          "difference": 0.25
        },
        {
          "group_id": "adapt_test_20",
          "Mixed512": 0.5,
          "A512": 0.0,
          "difference": 0.5
        },
        {
          "group_id": "adapt_test_21",
          "Mixed512": 0.5,
          "A512": 0.0,
          "difference": 0.5
        },
        {
          "group_id": "adapt_test_22",
          "Mixed512": 0.75,
          "A512": 0.0,
          "difference": 0.75
        },
        {
          "group_id": "adapt_test_23",
          "Mixed512": 0.75,
          "A512": 0.0,
          "difference": 0.75
        }
      ],
      "role": "named_secondary_replication"
    },
    "named_diagnostic": {
      "contrast": "Mixed512_minus_Offset",
      "renderer": "B",
      "condition": "correction_hard",
      "endpoint": "complete_four_case_quartet_accuracy",
      "groups": 24,
      "quartets": 192,
      "Mixed512": 0.6510416666666666,
      "Offset": 0.057291666666666664,
      "difference": 0.59375,
      "conditional_percentile_interval_95": [
        0.5104166666666666,
        0.671875
      ],
      "bootstrap_samples": 10000,
      "bootstrap_seed": 20261127,
      "uncertainty_scope": "Paired whole contexts conditional on frozen authored bank, old feature map, site and fitted heads; excludes coefficient/checkpoint uncertainty",
      "group_values": [
        {
          "group_id": "adapt_test_00",
          "Mixed512": 0.875,
          "Offset": 0.0,
          "difference": 0.875
        },
        {
          "group_id": "adapt_test_01",
          "Mixed512": 0.75,
          "Offset": 0.0,
          "difference": 0.75
        },
        {
          "group_id": "adapt_test_02",
          "Mixed512": 0.75,
          "Offset": 0.0,
          "difference": 0.75
        },
        {
          "group_id": "adapt_test_03",
          "Mixed512": 0.75,
          "Offset": 0.0,
          "difference": 0.75
        },
        {
          "group_id": "adapt_test_04",
          "Mixed512": 0.375,
          "Offset": 0.0,
          "difference": 0.375
        },
        {
          "group_id": "adapt_test_05",
          "Mixed512": 0.875,
          "Offset": 0.0,
          "difference": 0.875
        },
        {
          "group_id": "adapt_test_06",
          "Mixed512": 0.625,
          "Offset": 0.0,
          "difference": 0.625
        },
        {
          "group_id": "adapt_test_07",
          "Mixed512": 0.5,
          "Offset": 0.0,
          "difference": 0.5
        },
        {
          "group_id": "adapt_test_08",
          "Mixed512": 0.875,
          "Offset": 0.0,
          "difference": 0.875
        },
        {
          "group_id": "adapt_test_09",
          "Mixed512": 0.75,
          "Offset": 0.125,
          "difference": 0.625
        },
        {
          "group_id": "adapt_test_10",
          "Mixed512": 0.875,
          "Offset": 0.125,
          "difference": 0.75
        },
        {
          "group_id": "adapt_test_11",
          "Mixed512": 0.125,
          "Offset": 0.0,
          "difference": 0.125
        },
        {
          "group_id": "adapt_test_12",
          "Mixed512": 0.625,
          "Offset": 0.0,
          "difference": 0.625
        },
        {
          "group_id": "adapt_test_13",
          "Mixed512": 0.625,
          "Offset": 0.125,
          "difference": 0.5
        },
        {
          "group_id": "adapt_test_14",
          "Mixed512": 0.75,
          "Offset": 0.125,
          "difference": 0.625
        },
        {
          "group_id": "adapt_test_15",
          "Mixed512": 0.625,
          "Offset": 0.25,
          "difference": 0.375
        },
        {
          "group_id": "adapt_test_16",
          "Mixed512": 0.875,
          "Offset": 0.0,
          "difference": 0.875
        },
        {
          "group_id": "adapt_test_17",
          "Mixed512": 0.75,
          "Offset": 0.375,
          "difference": 0.375
        },
        {
          "group_id": "adapt_test_18",
          "Mixed512": 0.5,
          "Offset": 0.0,
          "difference": 0.5
        },
        {
          "group_id": "adapt_test_19",
          "Mixed512": 0.25,
          "Offset": 0.0,
          "difference": 0.25
        },
        {
          "group_id": "adapt_test_20",
          "Mixed512": 0.5,
          "Offset": 0.125,
          "difference": 0.375
        },
        {
          "group_id": "adapt_test_21",
          "Mixed512": 0.5,
          "Offset": 0.0,
          "difference": 0.5
        },
        {
          "group_id": "adapt_test_22",
          "Mixed512": 0.75,
          "Offset": 0.0,
          "difference": 0.75
        },
        {
          "group_id": "adapt_test_23",
          "Mixed512": 0.75,
          "Offset": 0.125,
          "difference": 0.625
        }
      ],
      "interpretation": "Difference between fixed fitting procedures at equal observed A/B budget; not a unique geometric or mechanistic attribution"
    }
  }
}

Reporting inputs (no new inference or analysis):
{
  "summary": {
    "path": "C:\\Users\\adirj\\OneDrive\\Documents\\GitHub\\6CCSAPRJ - Individual project\\Individual-Project\\docs\\results\\format-adaptation-2026-11-27\\summary.json",
    "sha256": "b20171e7c77ad704b37c29bec1aabe6bfcb5d7d60a38f4c5b9fe6132dc81392e"
  },
  "manifest": {
    "path": "C:\\Users\\adirj\\OneDrive\\Documents\\GitHub\\6CCSAPRJ - Individual project\\Individual-Project\\docs\\results\\format-adaptation-2026-11-27\\manifest.json",
    "sha256": "3d03d215a3f8e56abbe1aa76f708a072992caff4c33088f8bff32e90b3fce67a"
  },
  "audit": {
    "path": "C:\\Users\\adirj\\OneDrive\\Documents\\GitHub\\6CCSAPRJ - Individual project\\Individual-Project\\docs\\results\\format-adaptation-2026-11-27\\audit_receipt.json",
    "sha256": "1daa87e5692a6e04aacd46750b5919ba80e08dca94de7b2f3dcc492227fe660f"
  },
  "campaign_identity": {
    "path": "/home/adirj/kcl-latent-probes-runs/format_adaptation_v1_20261127_campaign",
    "sha256": null
  }
}

## Slide 3

The common frozen source-selected site, old scaler and regularization convention reduce adaptation confounds. They do not turn decodability into construct validity. The supplied four-factor rule is external to the model. A and B bundle system, narrative and question differences; this experiment does not isolate one prompt component. The new contextual groups share formal rules and an author pool. Local NF4 measurements do not establish BF16 robustness. Preserve the complete failed-control inventory and keep agent review distinct from human acceptance.

Reporting inputs (no new inference or analysis):
{
  "summary": {
    "path": "C:\\Users\\adirj\\OneDrive\\Documents\\GitHub\\6CCSAPRJ - Individual project\\Individual-Project\\docs\\results\\format-adaptation-2026-11-27\\summary.json",
    "sha256": "b20171e7c77ad704b37c29bec1aabe6bfcb5d7d60a38f4c5b9fe6132dc81392e"
  },
  "manifest": {
    "path": "C:\\Users\\adirj\\OneDrive\\Documents\\GitHub\\6CCSAPRJ - Individual project\\Individual-Project\\docs\\results\\format-adaptation-2026-11-27\\manifest.json",
    "sha256": "3d03d215a3f8e56abbe1aa76f708a072992caff4c33088f8bff32e90b3fce67a"
  },
  "audit": {
    "path": "C:\\Users\\adirj\\OneDrive\\Documents\\GitHub\\6CCSAPRJ - Individual project\\Individual-Project\\docs\\results\\format-adaptation-2026-11-27\\audit_receipt.json",
    "sha256": "1daa87e5692a6e04aacd46750b5919ba80e08dca94de7b2f3dcc492227fe660f"
  },
  "campaign_identity": {
    "path": "/home/adirj/kcl-latent-probes-runs/format_adaptation_v1_20261127_campaign",
    "sha256": null
  }
}

## Slide 4

The immediate deliverable is the empirical argument for BSPR and the dissertation, with supervisor agreement on scope. Later human review cannot retroactively make exposed outcomes confirmatory. A BF16 replication on CREATE is a proposal: freeze checkpoints, contexts, captures, heads, penalty convention and endpoint before another matched precision run. It has not been performed in this update. Broader communication or spontaneous-anticipation claims require a separately designed study, not a stronger label for these measurements. No scheduled date or additional experiment is claimed complete here.

Reporting inputs (no new inference or analysis):
{
  "summary": {
    "path": "C:\\Users\\adirj\\OneDrive\\Documents\\GitHub\\6CCSAPRJ - Individual project\\Individual-Project\\docs\\results\\format-adaptation-2026-11-27\\summary.json",
    "sha256": "b20171e7c77ad704b37c29bec1aabe6bfcb5d7d60a38f4c5b9fe6132dc81392e"
  },
  "manifest": {
    "path": "C:\\Users\\adirj\\OneDrive\\Documents\\GitHub\\6CCSAPRJ - Individual project\\Individual-Project\\docs\\results\\format-adaptation-2026-11-27\\manifest.json",
    "sha256": "3d03d215a3f8e56abbe1aa76f708a072992caff4c33088f8bff32e90b3fce67a"
  },
  "audit": {
    "path": "C:\\Users\\adirj\\OneDrive\\Documents\\GitHub\\6CCSAPRJ - Individual project\\Individual-Project\\docs\\results\\format-adaptation-2026-11-27\\audit_receipt.json",
    "sha256": "1daa87e5692a6e04aacd46750b5919ba80e08dca94de7b2f3dcc492227fe660f"
  },
  "campaign_identity": {
    "path": "/home/adirj/kcl-latent-probes-runs/format_adaptation_v1_20261127_campaign",
    "sha256": null
  }
}
