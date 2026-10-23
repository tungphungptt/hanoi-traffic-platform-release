# Bảng Đối Chiếu Sự Kiện Ground Truth & Dự Đoán (Event-Level Reconciliation Table)

## 1. Tổng Kết Phân Loại Lỗi & Bất Biến Số Học (Reconciliation Invariants)
- **Tổng số Ground Truth**: 96
- **Tổng số Dự đoán**: 107
- **Khớp chính xác (Exact Match / TP)**: 92 (95.8%)
- **Sai hướng rẽ (Wrong Movement)**: 0 (0.0%)
- **Sai loại xe (Wrong Class)**: 0 (0.0%)
- **Bỏ sót hoàn toàn (Pure False Negative)**: 4 (4.2%)
- **Dự đoán thừa ảo (Spurious False Positive)**: 8
- **Đếm lặp/trùng (Duplicate Count)**: 7

### Phương trình khớp số học (Exact Invariants):
- `Total_GT = 96` = `TP_Exact (92)` + `Wrong_Movement (0)` + `Wrong_Class (0)` + `Pure_FN (4)`
- `Total_Pred = 107` = `TP_Exact (92)` + `Wrong_Movement (0)` + `Wrong_Class (0)` + `Duplicate_Pred (7)` + `Spurious_FP (8)`
- `Temporal_Movement_FN = 4` = `Wrong_Movement (0)` + `Wrong_Class (0)` + `Pure_FN (4)`
- `Temporal_Movement_FP = 15` = `Wrong_Movement (0)` + `Duplicate_Pred (7)` + `Spurious_FP (8)`

## 2. Bảng Đối Chiếu 96 Sự Kiện Ground Truth
| GT ID | Frame GT | Time GT (s) | Mov GT | Cls GT | Pred ID | Frame Pred | Mov Pred | Cls Pred | Sai lệch Frame | Phân loại Lỗi |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | 15 | 1.5 | 3 | Car | 1 | 11 | 3 | Car | -4 | EXACT_MATCH |
| 2 | 19 | 1.9 | 4 | Car | 2 | 14 | 4 | Car | -5 | EXACT_MATCH |
| 3 | 33 | 3.3 | 2 | Car | 3 | 31 | 2 | Car | -2 | EXACT_MATCH |
| 4 | 37 | 3.7 | 3 | Car | 6 | 35 | 3 | Car | -2 | EXACT_MATCH |
| 5 | 45 | 4.5 | 3 | Car | 9 | 43 | 3 | Car | -2 | EXACT_MATCH |
| 6 | 47 | 4.7 | 2 | Car | 8 | 39 | 2 | Car | -8 | EXACT_MATCH |
| 7 | 47 | 4.7 | 3 | Car | 7 | 38 | 3 | Car | -9 | EXACT_MATCH |
| 8 | 49 | 4.9 | 4 | Car | 10 | 45 | 4 | Car | -4 | EXACT_MATCH |
| 9 | 65 | 6.5 | 3 | Car | 11 | 60 | 3 | Car | -5 | EXACT_MATCH |
| 10 | 69 | 6.9 | 3 | Car | 16 | 69 | 3 | Car | +0 | EXACT_MATCH |
| 11 | 71 | 7.1 | 4 | Car | 17 | 70 | 4 | Car | -1 | EXACT_MATCH |
| 12 | 75 | 7.5 | 2 | Car | 15 | 69 | 2 | Car | -6 | EXACT_MATCH |
| 13 | 75 | 7.5 | 3 | Car | 12 | 66 | 3 | Car | -9 | EXACT_MATCH |
| 14 | 81 | 8.1 | 3 | Car | 18 | 76 | 3 | Car | -5 | EXACT_MATCH |
| 15 | 93 | 9.3 | 2 | Car | 20 | 86 | 2 | Car | -7 | EXACT_MATCH |
| 16 | 98 | 9.8 | 1 | Car | - | - | - | - | - | FALSE_NEGATIVE |
| 17 | 105 | 10.5 | 3 | Car | 21 | 100 | 3 | Car | -5 | EXACT_MATCH |
| 18 | 109 | 10.9 | 3 | Car | 24 | 119 | 3 | Car | +10 | EXACT_MATCH |
| 19 | 113 | 11.3 | 4 | Car | 23 | 109 | 4 | Car | -4 | EXACT_MATCH |
| 20 | 117 | 11.7 | 2 | Car | 22 | 108 | 2 | Car | -9 | EXACT_MATCH |
| 21 | 123 | 12.3 | 3 | Car | 25 | 122 | 3 | Car | -1 | EXACT_MATCH |
| 22 | 129 | 12.9 | 3 | Car | - | - | - | - | - | FALSE_NEGATIVE |
| 23 | 139 | 13.9 | 10 | Car | - | - | - | - | - | FALSE_NEGATIVE |
| 24 | 143 | 14.3 | 3 | Car | 26 | 145 | 3 | Car | +2 | EXACT_MATCH |
| 25 | 185 | 18.5 | 4 | Car | 27 | 181 | 4 | Car | -4 | EXACT_MATCH |
| 26 | 207 | 20.7 | 11 | Car | 30 | 203 | 11 | Car | -4 | EXACT_MATCH |
| 27 | 219 | 21.9 | 12 | Car | 31 | 216 | 12 | Car | -3 | EXACT_MATCH |
| 28 | 229 | 22.9 | 11 | Car | 32 | 226 | 11 | Car | -3 | EXACT_MATCH |
| 29 | 233 | 23.3 | 11 | Car | 33 | 230 | 11 | Car | -3 | EXACT_MATCH |
| 30 | 241 | 24.1 | 4 | Car | 34 | 232 | 4 | Car | -9 | EXACT_MATCH |
| 31 | 245 | 24.5 | 11 | Car | 36 | 245 | 11 | Car | +0 | EXACT_MATCH |
| 32 | 249 | 24.9 | 11 | Car | 35 | 242 | 11 | Car | -7 | EXACT_MATCH |
| 33 | 261 | 26.1 | 11 | Car | 37 | 251 | 11 | Car | -10 | EXACT_MATCH |
| 34 | 267 | 26.7 | 11 | Car | 38 | 263 | 11 | Car | -4 | EXACT_MATCH |
| 35 | 277 | 27.7 | 11 | Car | 39 | 274 | 11 | Car | -3 | EXACT_MATCH |
| 36 | 277 | 27.7 | 4 | Car | 40 | 276 | 4 | Car | -1 | EXACT_MATCH |
| 37 | 281 | 28.1 | 11 | Car | 41 | 278 | 11 | Car | -3 | EXACT_MATCH |
| 38 | 289 | 28.9 | 11 | Car | 42 | 285 | 11 | Car | -4 | EXACT_MATCH |
| 39 | 295 | 29.5 | 11 | Car | 43 | 292 | 11 | Car | -3 | EXACT_MATCH |
| 40 | 305 | 30.5 | 11 | Car | 44 | 301 | 11 | Car | -4 | EXACT_MATCH |
| 41 | 311 | 31.1 | 11 | Car | 46 | 308 | 11 | Car | -3 | EXACT_MATCH |
| 42 | 311 | 31.1 | 4 | Car | 45 | 308 | 4 | Car | -3 | EXACT_MATCH |
| 43 | 317 | 31.7 | 11 | Car | 47 | 314 | 11 | Car | -3 | EXACT_MATCH |
| 44 | 325 | 32.5 | 11 | Car | 48 | 322 | 11 | Car | -3 | EXACT_MATCH |
| 45 | 331 | 33.1 | 11 | Car | 49 | 327 | 11 | Car | -4 | EXACT_MATCH |
| 46 | 339 | 33.9 | 5 | Car | 52 | 338 | 5 | Car | -1 | EXACT_MATCH |
| 47 | 343 | 34.3 | 5 | Car | 51 | 335 | 5 | Car | -8 | EXACT_MATCH |
| 48 | 347 | 34.7 | 11 | Car | 54 | 346 | 11 | Car | -1 | EXACT_MATCH |
| 49 | 349 | 34.9 | 11 | Car | 53 | 344 | 11 | Car | -5 | EXACT_MATCH |
| 50 | 359 | 35.9 | 5 | Car | 55 | 353 | 5 | Car | -6 | EXACT_MATCH |
| 51 | 361 | 36.1 | 5 | Car | 57 | 370 | 5 | Car | +9 | EXACT_MATCH |
| 52 | 369 | 36.9 | 11 | Car | 56 | 365 | 11 | Car | -4 | EXACT_MATCH |
| 53 | 375 | 37.5 | 5 | Car | 58 | 372 | 5 | Car | -3 | EXACT_MATCH |
| 54 | 385 | 38.5 | 11 | Car | 60 | 382 | 11 | Car | -3 | EXACT_MATCH |
| 55 | 399 | 39.9 | 5 | Car | 62 | 393 | 5 | Car | -6 | EXACT_MATCH |
| 56 | 411 | 41.1 | 11 | Car | 65 | 408 | 11 | Car | -3 | EXACT_MATCH |
| 57 | 413 | 41.3 | 11 | Truck | 63 | 402 | 11 | Truck | -11 | EXACT_MATCH |
| 58 | 417 | 41.7 | 5 | Car | 64 | 406 | 5 | Car | -11 | EXACT_MATCH |
| 59 | 425 | 42.5 | 11 | Car | 67 | 422 | 11 | Car | -3 | EXACT_MATCH |
| 60 | 425 | 42.5 | 5 | Truck | 68 | 424 | 5 | Truck | -1 | EXACT_MATCH |
| 61 | 429 | 42.9 | 5 | Car | 69 | 425 | 5 | Car | -4 | EXACT_MATCH |
| 62 | 433 | 43.3 | 11 | Car | 70 | 429 | 11 | Car | -4 | EXACT_MATCH |
| 63 | 441 | 44.1 | 11 | Car | 71 | 437 | 11 | Car | -4 | EXACT_MATCH |
| 64 | 445 | 44.5 | 5 | Car | 72 | 439 | 5 | Car | -6 | EXACT_MATCH |
| 65 | 451 | 45.1 | 10 | Car | 74 | 448 | 10 | Car | -3 | EXACT_MATCH |
| 66 | 451 | 45.1 | 11 | Car | 73 | 447 | 11 | Car | -4 | EXACT_MATCH |
| 67 | 453 | 45.3 | 5 | Truck | - | - | - | - | - | FALSE_NEGATIVE |
| 68 | 457 | 45.7 | 11 | Car | 75 | 453 | 11 | Car | -4 | EXACT_MATCH |
| 69 | 465 | 46.5 | 5 | Car | 77 | 465 | 5 | Car | +0 | EXACT_MATCH |
| 70 | 467 | 46.7 | 11 | Car | 76 | 463 | 11 | Car | -4 | EXACT_MATCH |
| 71 | 471 | 47.1 | 5 | Car | 81 | 476 | 5 | Car | +5 | EXACT_MATCH |
| 72 | 475 | 47.5 | 4 | Car | 78 | 470 | 4 | Car | -5 | EXACT_MATCH |
| 73 | 477 | 47.7 | 11 | Car | 80 | 474 | 11 | Car | -3 | EXACT_MATCH |
| 74 | 479 | 47.9 | 11 | Car | 79 | 473 | 11 | Car | -6 | EXACT_MATCH |
| 75 | 487 | 48.7 | 11 | Car | 83 | 485 | 11 | Car | -2 | EXACT_MATCH |
| 76 | 489 | 48.9 | 5 | Car | 82 | 483 | 5 | Car | -6 | EXACT_MATCH |
| 77 | 491 | 49.1 | 5 | Car | 85 | 497 | 5 | Car | +6 | EXACT_MATCH |
| 78 | 495 | 49.5 | 11 | Car | 84 | 492 | 11 | Car | -3 | EXACT_MATCH |
| 79 | 503 | 50.3 | 5 | Car | 87 | 508 | 5 | Car | +5 | EXACT_MATCH |
| 80 | 507 | 50.7 | 11 | Car | 86 | 504 | 11 | Car | -3 | EXACT_MATCH |
| 81 | 513 | 51.3 | 11 | Car | 88 | 508 | 11 | Car | -5 | EXACT_MATCH |
| 82 | 513 | 51.3 | 5 | Car | 89 | 521 | 5 | Car | +8 | EXACT_MATCH |
| 83 | 527 | 52.7 | 5 | Car | 90 | 533 | 5 | Car | +6 | EXACT_MATCH |
| 84 | 537 | 53.7 | 11 | Car | 91 | 533 | 11 | Car | -4 | EXACT_MATCH |
| 85 | 543 | 54.3 | 5 | Car | 92 | 538 | 5 | Car | -5 | EXACT_MATCH |
| 86 | 545 | 54.5 | 5 | Car | 94 | 551 | 5 | Car | +6 | EXACT_MATCH |
| 87 | 551 | 55.1 | 11 | Car | 93 | 548 | 11 | Car | -3 | EXACT_MATCH |
| 88 | 555 | 55.5 | 5 | Car | 95 | 558 | 5 | Car | +3 | EXACT_MATCH |
| 89 | 565 | 56.5 | 4 | Car | 96 | 559 | 4 | Car | -6 | EXACT_MATCH |
| 90 | 567 | 56.7 | 5 | Car | 97 | 565 | 5 | Car | -2 | EXACT_MATCH |
| 91 | 569 | 56.9 | 5 | Car | 99 | 577 | 5 | Car | +8 | EXACT_MATCH |
| 92 | 577 | 57.7 | 11 | Car | 98 | 575 | 11 | Car | -2 | EXACT_MATCH |
| 93 | 587 | 58.7 | 5 | Car | 100 | 584 | 5 | Car | -3 | EXACT_MATCH |
| 94 | 589 | 58.9 | 5 | Car | 103 | 599 | 5 | Car | +10 | EXACT_MATCH |
| 95 | 591 | 59.1 | 7 | Car | 101 | 585 | 7 | Car | -6 | EXACT_MATCH |
| 96 | 593 | 59.3 | 11 | Car | 102 | 591 | 11 | Car | -2 | EXACT_MATCH |

## 3. Danh Sách Các Dự Đoán Thừa / Đếm Lặp (Unmatched Predictions)
| Pred ID | Frame Pred | Time (s) | Mov Pred | Cls Pred | Trạng thái |
| :---: | :---: | :---: | :---: | :---: | :--- |
| 4 | 31 | 3.1 | 2 | Car | DUPLICATE_COUNT |
| 5 | 31 | 3.1 | 3 | Car | DUPLICATE_COUNT |
| 13 | 67 | 6.7 | 2 | Car | DUPLICATE_COUNT |
| 14 | 67 | 6.7 | 5 | Car | FALSE_POSITIVE |
| 19 | 80 | 8.0 | 5 | Car | FALSE_POSITIVE |
| 28 | 192 | 19.2 | 11 | Car | FALSE_POSITIVE |
| 29 | 193 | 19.3 | 4 | Car | FALSE_POSITIVE |
| 50 | 334 | 33.4 | 4 | Car | FALSE_POSITIVE |
| 59 | 380 | 38.0 | 7 | Truck | FALSE_POSITIVE |
| 61 | 386 | 38.6 | 4 | Car | FALSE_POSITIVE |
| 66 | 422 | 42.2 | 3 | Car | FALSE_POSITIVE |
| 104 | 599 | 59.9 | 5 | Car | DUPLICATE_COUNT |
| 105 | 599 | 59.9 | 5 | Car | DUPLICATE_COUNT |
| 106 | 599 | 59.9 | 10 | Car | DUPLICATE_COUNT |
| 107 | 599 | 59.9 | 10 | Car | DUPLICATE_COUNT |
