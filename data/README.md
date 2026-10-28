# Triage evaluation data

This folder holds the labelled data for the confidence-aware customer-support triage experiment. You can rebuild it from the repo root:

```bash
python -I eval/prepare_data.py
```

The script downloads raw files only if they are missing from `data/raw/`. It reports a warning if a raw file's SHA-256 differs from the published build. The script does not regenerate the hand-authored files (`queues.json`, `intent_map.json`, `clinc_oos_exclusions.json`, `reworded_questions.json`). It only reads and validates them.

## Sources and licences

| Dataset | Where we got it | Licence (checked against the source) |
|---|---|---|
| Banking77 (Casanueva et al., 2020): 13,083 online-banking customer queries, 77 intents (train 10,003 / test 3,080) | `train.csv` and `test.csv` from https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data. We could not use the Hugging Face parquet conversion for `PolyAI/banking77`: on 2026-10-14 both the API and datasets-server returned "Cannot get the config names for the dataset" because the dataset still uses a loading script. | CC BY 4.0. The repo `LICENSE` file is "Attribution 4.0 International", and the HF card metadata says `cc-by-4.0`. |
| CLINC150 / oos-eval (Larson et al., 2019): out-of-scope queries only | `data_full.json` from https://github.com/clinc/oos-eval (`oos_train` 100, `oos_val` 100, `oos_test` 1,000) | CC BY 3.0. The repo `LICENSE` file is "Attribution 3.0 Unported", and the HF card metadata (`clinc/clinc_oos`) says `cc-by-3.0`. |

Copies of the licence files and dataset cards are saved next to the raw data in `raw/banking77/` and `raw/clinc150/`. Neither dataset is modified, apart from the queue labels we add and the exclusions listed below.

## Queues and mapping method

`queues.json` lists the seven queues in a fixed order, each with a one- or two-sentence brief. We feed these briefs word for word to Jev as the option criteria. They never mention Banking77 intent names.

`intent_map.json` maps each of the 77 Banking77 intents to exactly one banking queue. No intent maps to `other`. It is the ground truth for the blog. We built it by hand as follows:

1. We read about six test examples per intent. Several intent names are misleading, so we mapped what the messages actually say. For example, `apple_pay_or_google_pay` is about topping up with a wallet, `get_physical_card` is about the PIN, `supported_cards_and_currencies` is about which cards and currencies can be used to add money, and `transfer_into_account` is about topping up by bank transfer.
2. We applied the following tie-break rules consistently, and wrote them into the queue briefs:
   - Any fee or charge question goes to `fees_exchange`, whatever it was charged on.
   - Any payment, direct debit or cash withdrawal the customer does not recognise goes to `account_security`.
   - Adding money to your own account goes to `top_up_balance`. Sending money to, or receiving it from, other people goes to `payments_transfers`.
   - A card declined at a merchant goes to `card_services`. A pending, reverted or duplicated card payment goes to `payments_transfers`.
3. `self_serve = true` means a standard help-centre article could fully resolve the request, with no staff investigation of, or action on, the account. 41 of the 77 intents are self-serve. Questions of the form "why was I charged a fee for X" count as self-serve, because the fee schedule explains them. Rows from `other` have `self_serve = null`.
4. `ambiguous = true` means a reasonable support manager could put the intent in a different queue. The `rationale` field names the alternative queue. **20 intents are ambiguous**, which is more than the 8-15 we first expected. Most of them are fee questions about payments, transfers, top-ups or ATM withdrawals, and "not recognised" intents that sit between security and payments. They are: apple_pay_or_google_pay, balance_not_updated_after_bank_transfer, card_payment_fee_charged, card_payment_not_recognised, card_swallowed, cash_withdrawal_charge, cash_withdrawal_not_recognised, country_support, declined_card_payment, direct_debit_payment_not_recognised, extra_charge_on_statement, lost_or_stolen_card, supported_cards_and_currencies, top_up_by_bank_transfer_charge, top_up_by_card_charge, transaction_charged_twice, transfer_fee_charged, transfer_into_account, verify_top_up, wrong_exchange_rate_for_cash_withdrawal.

Intents per queue: card_services 20, payments_transfers 13, top_up_balance 13, cash_atm 5, fees_exchange 12, account_security 14.

### CLINC out-of-scope exclusions

Some CLINC150 "out-of-scope" queries are banking requests, for example "can i deposit a check", "how long do wire transfers take" and "i need to update my address". A banking support agent could reasonably route these to a banking queue, so labelling them `other` would add noise. We scanned `oos_test` and `oos_train` for banking keywords and reviewed the hits by hand. We removed 31 `oos_test` texts and 4 `oos_train` texts before sampling. They are listed in `clinc_oos_exclusions.json`. Every other OOS row gets `ambiguous = false`. The keyword scan may have missed some borderline texts, such as "can my grace period be extended any possible way" in `eval.jsonl`.

## Files

Every JSONL row has the fields `id`, `text`, `source` (`banking77` or `clinc_oos`), `intent` (the Banking77 intent, or `oos`), `queue`, `self_serve` and `ambiguous`. IDs point back to the row in the raw file: `b77-<split>-<row index>` or `clinc-<split>-<index>`.

| File | What it contains |
|---|---|
| `eval.jsonl` | Banking77 **test**, 10 per intent, stratified (770 rows), plus 100 CLINC `oos_test` rows labelled `other`. Shuffled. 870 rows. |
| `eval_small.jsonl` | 140-row subset of `eval.jsonl`: 20 per queue, including `other`, spread round-robin across intents. Used for the repeated robustness runs (option shuffle and rewording). |
| `train_fewshot20.jsonl` | Banking77 **train**, 20 per banking queue, spread round-robin across that queue's intents, plus 20 CLINC `oos_train` rows. 140 rows. |
| `train_full.jsonl` | All Banking77 train rows mapped to queues, plus all remaining (96) `oos_train` rows. 10,099 rows. |
| `reworded_questions.json` | `instructions` holds the base question ("Which support queue should handle this customer message?") and two rewordings. `queues_alt` holds one paraphrase per queue brief with the same meaning. |
| `stats.json` | The counts below, written by the build script. |

Rows per queue:

| Queue | eval | eval_small | train_fewshot20 | train_full |
|---|---:|---:|---:|---:|
| card_services | 200 | 20 | 20 | 2,264 |
| payments_transfers | 130 | 20 | 20 | 1,951 |
| top_up_balance | 130 | 20 | 20 | 1,727 |
| cash_atm | 50 | 20 | 20 | 644 |
| fees_exchange | 120 | 20 | 20 | 1,734 |
| account_security | 140 | 20 | 20 | 1,683 |
| other | 100 | 20 | 20 | 96 |
| **total** | **870** | **140** | **140** | **10,099** |

`eval.jsonl` has 200 rows from ambiguous intents and 410 self-serve rows. No text appears twice in `eval.jsonl`, and no `eval.jsonl` text appears in `train_full.jsonl` (case-insensitive exact match).

Because each queue in `eval.jsonl` gets 10 rows per intent, the queues are imbalanced. `cash_atm` has 50 rows and `card_services` has 200. Report macro-F1 alongside accuracy.

## Seed

All sampling uses Python's `random.Random(7)`. Each output file gets a fresh `Random(7)` instance, and inputs are sorted deterministically before sampling. Running the script twice produces byte-identical files.
