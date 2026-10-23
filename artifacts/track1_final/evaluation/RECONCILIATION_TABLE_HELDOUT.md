# Bảng Đối Chiếu Sự Kiện Ground Truth & Dự Đoán (Event-Level Reconciliation Table)

## 1. Tổng Kết Phân Loại Lỗi & Bất Biến Số Học (Reconciliation Invariants)
- **Tổng số Ground Truth**: 47
- **Tổng số Dự đoán**: 53
- **Khớp chính xác (Exact Match / TP)**: 41 (87.2%)
- **Sai hướng rẽ (Wrong Movement)**: 5 (10.6%)
- **Sai loại xe (Wrong Class)**: 0 (0.0%)
- **Bỏ sót hoàn toàn (Pure False Negative)**: 1 (2.1%)
- **Dự đoán thừa ảo (Spurious False Positive)**: 3
- **Đếm lặp/trùng (Duplicate Count)**: 4

### Phương trình khớp số học (Exact Invariants):
- `Total_GT = 47` = `TP_Exact (41)` + `Wrong_Movement (5)` + `Wrong_Class (0)` + `Pure_FN (1)`
- `Total_Pred = 53` = `TP_Exact (41)` + `Wrong_Movement (5)` + `Wrong_Class (0)` + `Duplicate_Pred (4)` + `Spurious_FP (3)`
- `Temporal_Movement_FN = 6` = `Wrong_Movement (5)` + `Wrong_Class (0)` + `Pure_FN (1)`
- `Temporal_Movement_FP = 12` = `Wrong_Movement (5)` + `Duplicate_Pred (4)` + `Spurious_FP (3)`

## 2. Bảng Đối Chiếu 96 Sự Kiện Ground Truth
| GT ID | Frame GT | Time GT (s) | Mov GT | Cls GT | Pred ID | Frame Pred | Mov Pred | Cls Pred | Sai lệch Frame | Phân loại Lỗi |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | 359 | 35.9 | 5 | Car | 1 | 353 | 5 | Car | -6 | EXACT_MATCH |
| 2 | 361 | 36.1 | 5 | Car | 4 | 372 | 4 | Car | +11 | WRONG_MOVEMENT |
| 3 | 369 | 36.9 | 11 | Car | 2 | 365 | 11 | Car | -4 | EXACT_MATCH |
| 4 | 375 | 37.5 | 5 | Car | 3 | 370 | 5 | Car | -5 | EXACT_MATCH |
| 5 | 385 | 38.5 | 11 | Car | 6 | 382 | 11 | Car | -3 | EXACT_MATCH |
| 6 | 399 | 39.9 | 5 | Car | 8 | 393 | 5 | Car | -6 | EXACT_MATCH |
| 7 | 411 | 41.1 | 11 | Car | 11 | 408 | 11 | Car | -3 | EXACT_MATCH |
| 8 | 413 | 41.3 | 11 | Truck | 9 | 402 | 3 | Truck | -11 | WRONG_MOVEMENT |
| 9 | 417 | 41.7 | 5 | Car | 10 | 406 | 5 | Car | -11 | EXACT_MATCH |
| 10 | 425 | 42.5 | 11 | Car | 13 | 422 | 11 | Car | -3 | EXACT_MATCH |
| 11 | 425 | 42.5 | 5 | Truck | 14 | 424 | 4 | Truck | -1 | WRONG_MOVEMENT |
| 12 | 429 | 42.9 | 5 | Car | 15 | 425 | 5 | Car | -4 | EXACT_MATCH |
| 13 | 433 | 43.3 | 11 | Car | 16 | 429 | 11 | Car | -4 | EXACT_MATCH |
| 14 | 441 | 44.1 | 11 | Car | 17 | 437 | 11 | Car | -4 | EXACT_MATCH |
| 15 | 445 | 44.5 | 5 | Car | 18 | 439 | 5 | Car | -6 | EXACT_MATCH |
| 16 | 451 | 45.1 | 10 | Car | 20 | 448 | 4 | Car | -3 | WRONG_MOVEMENT |
| 17 | 451 | 45.1 | 11 | Car | 19 | 447 | 11 | Car | -4 | EXACT_MATCH |
| 18 | 453 | 45.3 | 5 | Truck | - | - | - | - | - | FALSE_NEGATIVE |
| 19 | 457 | 45.7 | 11 | Car | 21 | 453 | 11 | Car | -4 | EXACT_MATCH |
| 20 | 465 | 46.5 | 5 | Car | 23 | 465 | 1 | Car | +0 | WRONG_MOVEMENT |
| 21 | 467 | 46.7 | 11 | Car | 22 | 463 | 11 | Car | -4 | EXACT_MATCH |
| 22 | 471 | 47.1 | 5 | Car | 27 | 476 | 5 | Car | +5 | EXACT_MATCH |
| 23 | 475 | 47.5 | 4 | Car | 24 | 470 | 4 | Car | -5 | EXACT_MATCH |
| 24 | 477 | 47.7 | 11 | Car | 26 | 474 | 11 | Car | -3 | EXACT_MATCH |
| 25 | 479 | 47.9 | 11 | Car | 25 | 473 | 11 | Car | -6 | EXACT_MATCH |
| 26 | 487 | 48.7 | 11 | Car | 29 | 485 | 11 | Car | -2 | EXACT_MATCH |
| 27 | 489 | 48.9 | 5 | Car | 28 | 483 | 5 | Car | -6 | EXACT_MATCH |
| 28 | 491 | 49.1 | 5 | Car | 31 | 497 | 5 | Car | +6 | EXACT_MATCH |
| 29 | 495 | 49.5 | 11 | Car | 30 | 492 | 11 | Car | -3 | EXACT_MATCH |
| 30 | 503 | 50.3 | 5 | Car | 33 | 508 | 5 | Car | +5 | EXACT_MATCH |
| 31 | 507 | 50.7 | 11 | Car | 32 | 504 | 11 | Car | -3 | EXACT_MATCH |
| 32 | 513 | 51.3 | 11 | Car | 34 | 508 | 11 | Car | -5 | EXACT_MATCH |
| 33 | 513 | 51.3 | 5 | Car | 35 | 521 | 5 | Car | +8 | EXACT_MATCH |
| 34 | 527 | 52.7 | 5 | Car | 36 | 533 | 5 | Car | +6 | EXACT_MATCH |
| 35 | 537 | 53.7 | 11 | Car | 37 | 533 | 11 | Car | -4 | EXACT_MATCH |
| 36 | 543 | 54.3 | 5 | Car | 38 | 538 | 5 | Car | -5 | EXACT_MATCH |
| 37 | 545 | 54.5 | 5 | Car | 40 | 551 | 5 | Car | +6 | EXACT_MATCH |
| 38 | 551 | 55.1 | 11 | Car | 39 | 548 | 11 | Car | -3 | EXACT_MATCH |
| 39 | 555 | 55.5 | 5 | Car | 41 | 558 | 5 | Car | +3 | EXACT_MATCH |
| 40 | 565 | 56.5 | 4 | Car | 42 | 559 | 4 | Car | -6 | EXACT_MATCH |
| 41 | 567 | 56.7 | 5 | Car | 43 | 565 | 5 | Car | -2 | EXACT_MATCH |
| 42 | 569 | 56.9 | 5 | Car | 45 | 577 | 5 | Car | +8 | EXACT_MATCH |
| 43 | 577 | 57.7 | 11 | Car | 44 | 575 | 11 | Car | -2 | EXACT_MATCH |
| 44 | 587 | 58.7 | 5 | Car | 46 | 584 | 5 | Car | -3 | EXACT_MATCH |
| 45 | 589 | 58.9 | 5 | Car | 52 | 599 | 5 | Car | +10 | EXACT_MATCH |
| 46 | 591 | 59.1 | 7 | Car | 47 | 585 | 7 | Car | -6 | EXACT_MATCH |
| 47 | 593 | 59.3 | 11 | Car | 48 | 591 | 11 | Car | -2 | EXACT_MATCH |

## 3. Danh Sách Các Dự Đoán Thừa / Đếm Lặp (Unmatched Predictions)
| Pred ID | Frame Pred | Time (s) | Mov Pred | Cls Pred | Trạng thái |
| :---: | :---: | :---: | :---: | :---: | :--- |
| 5 | 380 | 38.0 | 4 | Truck | FALSE_POSITIVE |
| 7 | 386 | 38.6 | 4 | Car | FALSE_POSITIVE |
| 12 | 422 | 42.2 | 3 | Car | FALSE_POSITIVE |
| 49 | 599 | 59.9 | 4 | Car | DUPLICATE_COUNT |
| 50 | 599 | 59.9 | 4 | Car | DUPLICATE_COUNT |
| 51 | 599 | 59.9 | 4 | Car | DUPLICATE_COUNT |
| 53 | 599 | 59.9 | 5 | Car | DUPLICATE_COUNT |
