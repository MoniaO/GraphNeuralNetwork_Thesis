# Stage A backbone decision table — 11.08.2026

- Scenario screen: `clean`
- Progress: **288 ok / 0 fail / 288 planned** — COMPLETE
- Selection metric: **mean valid AUPRC** over seeds (test sealed, not ranked)
- Next stop: pick Top-1 (or Top-2) config **per backbone** before Stage C stats layer

## Overall ranking (valid AUPRC)

| rank | backbone | config_id | n_seeds | mean valid AUPRC ± std | mean valid AUC ± std | sealed test AUPRC (info only) |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | hgt | `hgt__h32__L2__d0.25__lr0.001__hd4__shared` | 3 | 0.7288 ± 0.0091 | 0.8572 ± 0.0060 | 0.5546 ± 0.0161 |
| 2 | hgt | `hgt__h32__L4__d0.25__lr0.001__hd4__shared` | 3 | 0.7262 ± 0.0334 | 0.8606 ± 0.0165 | 0.5535 ± 0.0078 |
| 3 | hgt | `hgt__h32__L2__d0.1__lr0.0003__hd4__shared` | 3 | 0.7253 ± 0.0248 | 0.8617 ± 0.0071 | 0.5762 ± 0.0154 |
| 4 | hgt | `hgt__h32__L2__d0.1__lr0.001__hd4__shared` | 3 | 0.7249 ± 0.0081 | 0.8570 ± 0.0053 | 0.5674 ± 0.0083 |
| 5 | hgt | `hgt__h64__L2__d0.25__lr0.0003__hd4__shared` | 3 | 0.7197 ± 0.0072 | 0.8540 ± 0.0041 | 0.5541 ± 0.0047 |
| 6 | hgt | `hgt__h64__L3__d0.25__lr0.0003__hd4__shared` | 3 | 0.7183 ± 0.0063 | 0.8578 ± 0.0021 | 0.5635 ± 0.0080 |
| 7 | hgt | `hgt__h32__L4__d0.1__lr0.001__hd4__shared` | 3 | 0.7170 ± 0.0335 | 0.8590 ± 0.0110 | 0.5752 ± 0.0171 |
| 8 | hgt | `hgt__h32__L4__d0.25__lr0.0003__hd4__shared` | 3 | 0.7136 ± 0.0442 | 0.8585 ± 0.0156 | 0.5740 ± 0.0154 |
| 9 | hgt | `hgt__h32__L4__d0.1__lr0.0003__hd4__shared` | 3 | 0.7118 ± 0.0285 | 0.8585 ± 0.0135 | 0.5761 ± 0.0141 |
| 10 | hgt | `hgt__h64__L3__d0.1__lr0.0003__hd4__shared` | 3 | 0.7117 ± 0.0082 | 0.8597 ± 0.0032 | 0.5634 ± 0.0142 |
| 11 | hgt | `hgt__h64__L2__d0.1__lr0.0003__hd4__shared` | 3 | 0.7091 ± 0.0131 | 0.8530 ± 0.0067 | 0.5634 ± 0.0080 |
| 12 | hgt | `hgt__h64__L4__d0.25__lr0.001__hd4__shared` | 3 | 0.7082 ± 0.0055 | 0.8515 ± 0.0014 | 0.5531 ± 0.0231 |
| 13 | hgt | `hgt__h64__L2__d0.25__lr0.001__hd4__shared` | 3 | 0.7081 ± 0.0050 | 0.8447 ± 0.0198 | 0.5518 ± 0.0237 |
| 14 | hgt | `hgt__h64__L2__d0.1__lr0.001__hd4__shared` | 3 | 0.7060 ± 0.0091 | 0.8511 ± 0.0041 | 0.5608 ± 0.0043 |
| 15 | hgt | `hgt__h64__L3__d0.25__lr0.001__hd4__shared` | 3 | 0.7050 ± 0.0070 | 0.8532 ± 0.0053 | 0.5652 ± 0.0144 |
| 16 | hgt | `hgt__h32__L3__d0.1__lr0.0003__hd4__shared` | 3 | 0.7023 ± 0.0258 | 0.8463 ± 0.0089 | 0.5804 ± 0.0103 |
| 17 | hgt | `hgt__h64__L4__d0.1__lr0.001__hd4__shared` | 3 | 0.7007 ± 0.0050 | 0.8483 ± 0.0023 | 0.5666 ± 0.0192 |
| 18 | hgt | `hgt__h64__L4__d0.1__lr0.0003__hd4__shared` | 3 | 0.7004 ± 0.0092 | 0.8504 ± 0.0044 | 0.5640 ± 0.0033 |
| 19 | hgt | `hgt__h64__L3__d0.1__lr0.001__hd4__shared` | 3 | 0.7002 ± 0.0125 | 0.8545 ± 0.0129 | 0.5629 ± 0.0172 |
| 20 | hgt | `hgt__h64__L4__d0.25__lr0.0003__hd4__shared` | 3 | 0.7002 ± 0.0041 | 0.8484 ± 0.0050 | 0.5591 ± 0.0217 |
| 21 | hgt | `hgt__h32__L3__d0.1__lr0.001__hd4__shared` | 3 | 0.6998 ± 0.0085 | 0.8371 ± 0.0130 | 0.5797 ± 0.0052 |
| 22 | hgt | `hgt__h32__L3__d0.25__lr0.0003__hd4__shared` | 3 | 0.6990 ± 0.0165 | 0.8511 ± 0.0070 | 0.5826 ± 0.0125 |
| 23 | rgcn_matched | `rgcn_matched__h64__L3__d0.25__lr0.0003__b8__shared` | 3 | 0.6972 ± 0.0098 | 0.8362 ± 0.0187 | 0.5738 ± 0.0427 |
| 24 | hgt | `hgt__h32__L3__d0.25__lr0.001__hd4__shared` | 3 | 0.6968 ± 0.0186 | 0.8520 ± 0.0015 | 0.5615 ± 0.0284 |
| 25 | rgcn_matched | `rgcn_matched__h32__L2__d0.25__lr0.0003__b8__shared` | 3 | 0.6933 ± 0.0185 | 0.8383 ± 0.0075 | 0.5796 ± 0.0200 |
| 26 | rgcn_matched | `rgcn_matched__h32__L3__d0.25__lr0.001__b8__shared` | 3 | 0.6927 ± 0.0091 | 0.8404 ± 0.0152 | 0.5809 ± 0.0053 |
| 27 | rgcn_matched | `rgcn_matched__h32__L4__d0.25__lr0.001__b8__shared` | 3 | 0.6918 ± 0.0296 | 0.8345 ± 0.0098 | 0.5534 ± 0.0160 |
| 28 | hgt | `hgt__h32__L2__d0.25__lr0.0003__hd4__shared` | 3 | 0.6910 ± 0.0433 | 0.8142 ± 0.0791 | 0.5086 ± 0.1269 |
| 29 | rgcn_matched | `rgcn_matched__h32__L2__d0.25__lr0.001__b8__shared` | 3 | 0.6858 ± 0.0189 | 0.8320 ± 0.0119 | 0.5809 ± 0.0438 |
| 30 | rgcn_matched | `rgcn_matched__h32__L3__d0.25__lr0.0003__b8__shared` | 3 | 0.6857 ± 0.0138 | 0.8400 ± 0.0147 | 0.5661 ± 0.0050 |
| 31 | rgcn_matched | `rgcn_matched__h64__L3__d0.25__lr0.001__b8__shared` | 3 | 0.6846 ± 0.0095 | 0.8436 ± 0.0100 | 0.5629 ± 0.0168 |
| 32 | rgcn_matched | `rgcn_matched__h32__L4__d0.25__lr0.0003__b8__shared` | 3 | 0.6817 ± 0.0240 | 0.8375 ± 0.0038 | 0.5600 ± 0.0222 |
| 33 | rgcn_matched | `rgcn_matched__h64__L3__d0.1__lr0.001__b8__shared` | 3 | 0.6787 ± 0.0250 | 0.8285 ± 0.0143 | 0.5513 ± 0.0436 |
| 34 | rgcn_matched | `rgcn_matched__h64__L4__d0.25__lr0.0003__b8__shared` | 3 | 0.6746 ± 0.0260 | 0.8090 ± 0.0285 | 0.5228 ± 0.0622 |
| 35 | rgcn_matched | `rgcn_matched__h64__L2__d0.25__lr0.0003__b8__shared` | 3 | 0.6705 ± 0.0228 | 0.8277 ± 0.0020 | 0.5832 ± 0.0236 |
| 36 | hetero_sage_matched | `hetero_sage_matched__h32__L2__d0.25__lr0.0003__shared` | 3 | 0.6705 ± 0.0160 | 0.8041 ± 0.0080 | 0.5123 ± 0.0273 |
| 37 | rgcn_matched | `rgcn_matched__h32__L3__d0.1__lr0.001__b8__shared` | 3 | 0.6668 ± 0.0190 | 0.8184 ± 0.0041 | 0.5608 ± 0.0147 |
| 38 | rgcn_matched | `rgcn_matched__h64__L3__d0.1__lr0.0003__b8__shared` | 3 | 0.6662 ± 0.0152 | 0.8290 ± 0.0189 | 0.5809 ± 0.0397 |
| 39 | rgcn_matched | `rgcn_matched__h64__L2__d0.25__lr0.001__b8__shared` | 3 | 0.6661 ± 0.0130 | 0.8249 ± 0.0106 | 0.5735 ± 0.0121 |
| 40 | rgcn_matched | `rgcn_matched__h32__L2__d0.1__lr0.0003__b8__shared` | 3 | 0.6657 ± 0.0269 | 0.8282 ± 0.0054 | 0.5518 ± 0.0169 |
| 41 | rgcn_matched | `rgcn_matched__h32__L2__d0.1__lr0.001__b8__shared` | 3 | 0.6651 ± 0.0166 | 0.8206 ± 0.0043 | 0.5485 ± 0.0206 |
| 42 | rgcn_matched | `rgcn_matched__h32__L3__d0.1__lr0.0003__b8__shared` | 3 | 0.6649 ± 0.0181 | 0.8198 ± 0.0094 | 0.5395 ± 0.0235 |
| 43 | rgcn_matched | `rgcn_matched__h64__L4__d0.1__lr0.0003__b8__shared` | 3 | 0.6640 ± 0.0336 | 0.7912 ± 0.0250 | 0.5176 ± 0.0486 |
| 44 | rgcn_matched | `rgcn_matched__h64__L4__d0.25__lr0.001__b8__shared` | 3 | 0.6623 ± 0.0257 | 0.8067 ± 0.0337 | 0.5141 ± 0.0562 |
| 45 | rgcn_matched | `rgcn_matched__h64__L2__d0.1__lr0.0003__b8__shared` | 3 | 0.6617 ± 0.0046 | 0.8129 ± 0.0082 | 0.5625 ± 0.0294 |
| 46 | rgcn_matched | `rgcn_matched__h64__L4__d0.1__lr0.001__b8__shared` | 3 | 0.6589 ± 0.0254 | 0.7879 ± 0.0110 | 0.5126 ± 0.0119 |
| 47 | rgcn_matched | `rgcn_matched__h64__L2__d0.1__lr0.001__b8__shared` | 3 | 0.6588 ± 0.0069 | 0.8112 ± 0.0177 | 0.5758 ± 0.0410 |
| 48 | rgcn_matched | `rgcn_matched__h32__L4__d0.1__lr0.001__b8__shared` | 3 | 0.6548 ± 0.0285 | 0.8108 ± 0.0053 | 0.5451 ± 0.0538 |
| 49 | rgcn_matched | `rgcn_matched__h32__L4__d0.1__lr0.0003__b8__shared` | 3 | 0.6499 ± 0.0245 | 0.8146 ± 0.0075 | 0.5376 ± 0.0546 |
| 50 | hetero_gatv2 | `hetero_gatv2__h32__L2__d0.25__lr0.0003__hd4__shared` | 3 | 0.6497 ± 0.0057 | 0.7866 ± 0.0032 | 0.4951 ± 0.0424 |
| 51 | hetero_sage_matched | `hetero_sage_matched__h32__L2__d0.25__lr0.001__shared` | 3 | 0.6485 ± 0.0568 | 0.7922 ± 0.0230 | 0.5464 ± 0.0110 |
| 52 | hetero_gatv2 | `hetero_gatv2__h64__L2__d0.25__lr0.001__hd4__shared` | 3 | 0.6393 ± 0.0157 | 0.7936 ± 0.0048 | 0.5367 ± 0.0196 |
| 53 | hetero_sage_matched | `hetero_sage_matched__h32__L2__d0.1__lr0.001__shared` | 3 | 0.6347 ± 0.0195 | 0.7807 ± 0.0186 | 0.5659 ± 0.0473 |
| 54 | hetero_gatv2 | `hetero_gatv2__h32__L3__d0.25__lr0.001__hd4__shared` | 3 | 0.6277 ± 0.0187 | 0.7737 ± 0.0098 | 0.5025 ± 0.0432 |
| 55 | hetero_gatv2 | `hetero_gatv2__h64__L2__d0.25__lr0.0003__hd4__shared` | 3 | 0.6249 ± 0.0263 | 0.7699 ± 0.0169 | 0.4947 ± 0.0466 |
| 56 | hetero_gatv2 | `hetero_gatv2__h32__L2__d0.25__lr0.001__hd4__shared` | 3 | 0.6236 ± 0.0269 | 0.7986 ± 0.0190 | 0.5478 ± 0.0222 |
| 57 | hetero_sage_matched | `hetero_sage_matched__h64__L2__d0.1__lr0.0003__shared` | 3 | 0.6235 ± 0.0147 | 0.7856 ± 0.0124 | 0.4838 ± 0.0298 |
| 58 | hetero_sage_matched | `hetero_sage_matched__h32__L3__d0.25__lr0.0003__shared` | 3 | 0.6221 ± 0.0247 | 0.7758 ± 0.0083 | 0.5064 ± 0.0180 |
| 59 | hetero_gatv2 | `hetero_gatv2__h64__L2__d0.1__lr0.001__hd4__shared` | 3 | 0.6202 ± 0.0301 | 0.7690 ± 0.0153 | 0.4592 ± 0.0316 |
| 60 | hetero_gatv2 | `hetero_gatv2__h32__L2__d0.1__lr0.0003__hd4__shared` | 3 | 0.6189 ± 0.0198 | 0.7746 ± 0.0074 | 0.5042 ± 0.0607 |
| 61 | hetero_sage_matched | `hetero_sage_matched__h64__L2__d0.25__lr0.001__shared` | 3 | 0.6164 ± 0.0082 | 0.7928 ± 0.0176 | 0.5202 ± 0.0204 |
| 62 | hetero_gatv2 | `hetero_gatv2__h32__L4__d0.25__lr0.001__hd4__shared` | 3 | 0.6141 ± 0.0345 | 0.7671 ± 0.0212 | 0.5191 ± 0.0113 |
| 63 | hetero_sage_matched | `hetero_sage_matched__h64__L4__d0.1__lr0.0003__shared` | 3 | 0.6140 ± 0.0142 | 0.7618 ± 0.0101 | 0.4847 ± 0.0473 |
| 64 | hetero_gatv2 | `hetero_gatv2__h64__L4__d0.25__lr0.001__hd4__shared` | 3 | 0.6135 ± 0.0235 | 0.7655 ± 0.0159 | 0.5101 ± 0.0146 |
| 65 | hetero_gatv2 | `hetero_gatv2__h32__L3__d0.1__lr0.0003__hd4__shared` | 3 | 0.6125 ± 0.0223 | 0.7652 ± 0.0024 | 0.4989 ± 0.0160 |
| 66 | hetero_sage_matched | `hetero_sage_matched__h32__L3__d0.25__lr0.001__shared` | 3 | 0.6102 ± 0.0123 | 0.7648 ± 0.0119 | 0.4901 ± 0.0401 |
| 67 | hetero_gatv2 | `hetero_gatv2__h64__L3__d0.25__lr0.001__hd4__shared` | 3 | 0.6088 ± 0.0271 | 0.7616 ± 0.0079 | 0.4671 ± 0.0335 |
| 68 | hetero_gatv2 | `hetero_gatv2__h64__L3__d0.25__lr0.0003__hd4__shared` | 3 | 0.6085 ± 0.0185 | 0.7643 ± 0.0035 | 0.5179 ± 0.0326 |
| 69 | hetero_sage_matched | `hetero_sage_matched__h32__L2__d0.1__lr0.0003__shared` | 3 | 0.6082 ± 0.0069 | 0.7784 ± 0.0196 | 0.5231 ± 0.0681 |
| 70 | hetero_sage_matched | `hetero_sage_matched__h64__L2__d0.25__lr0.0003__shared` | 3 | 0.6068 ± 0.0415 | 0.7808 ± 0.0213 | 0.4908 ± 0.0181 |
| 71 | hetero_sage_matched | `hetero_sage_matched__h64__L2__d0.1__lr0.001__shared` | 3 | 0.6061 ± 0.0041 | 0.7801 ± 0.0071 | 0.4648 ± 0.0137 |
| 72 | hetero_sage_matched | `hetero_sage_matched__h64__L3__d0.25__lr0.0003__shared` | 3 | 0.6047 ± 0.0276 | 0.7666 ± 0.0035 | 0.4796 ± 0.0298 |
| 73 | hetero_sage_matched | `hetero_sage_matched__h64__L3__d0.1__lr0.001__shared` | 3 | 0.6047 ± 0.0065 | 0.7627 ± 0.0061 | 0.4401 ± 0.0715 |
| 74 | hetero_gatv2 | `hetero_gatv2__h32__L3__d0.25__lr0.0003__hd4__shared` | 3 | 0.6026 ± 0.0047 | 0.7571 ± 0.0218 | 0.4616 ± 0.0226 |
| 75 | hetero_gatv2 | `hetero_gatv2__h64__L2__d0.1__lr0.0003__hd4__shared` | 3 | 0.6006 ± 0.0139 | 0.7617 ± 0.0132 | 0.4912 ± 0.0051 |
| 76 | hetero_sage_matched | `hetero_sage_matched__h64__L3__d0.1__lr0.0003__shared` | 3 | 0.5998 ± 0.0153 | 0.7593 ± 0.0053 | 0.4981 ± 0.0072 |
| 77 | hetero_gatv2 | `hetero_gatv2__h64__L3__d0.1__lr0.0003__hd4__shared` | 3 | 0.5992 ± 0.0262 | 0.7518 ± 0.0144 | 0.4768 ± 0.0326 |
| 78 | hetero_sage_matched | `hetero_sage_matched__h32__L4__d0.25__lr0.001__shared` | 3 | 0.5978 ± 0.0135 | 0.7681 ± 0.0063 | 0.4656 ± 0.0407 |
| 79 | hetero_gatv2 | `hetero_gatv2__h32__L3__d0.1__lr0.001__hd4__shared` | 3 | 0.5960 ± 0.0204 | 0.7609 ± 0.0249 | 0.4783 ± 0.0396 |
| 80 | hetero_sage_matched | `hetero_sage_matched__h32__L3__d0.1__lr0.001__shared` | 3 | 0.5944 ± 0.0229 | 0.7679 ± 0.0217 | 0.4993 ± 0.0170 |
| 81 | hetero_gatv2 | `hetero_gatv2__h64__L4__d0.25__lr0.0003__hd4__shared` | 3 | 0.5941 ± 0.0097 | 0.7550 ± 0.0109 | 0.4968 ± 0.0071 |
| 82 | hetero_sage_matched | `hetero_sage_matched__h64__L4__d0.25__lr0.001__shared` | 3 | 0.5937 ± 0.0411 | 0.7688 ± 0.0101 | 0.4810 ± 0.0341 |
| 83 | hetero_sage_matched | `hetero_sage_matched__h64__L3__d0.25__lr0.001__shared` | 3 | 0.5917 ± 0.0345 | 0.7590 ± 0.0296 | 0.4625 ± 0.0666 |
| 84 | hetero_sage_matched | `hetero_sage_matched__h32__L4__d0.1__lr0.0003__shared` | 3 | 0.5887 ± 0.0272 | 0.7532 ± 0.0127 | 0.4547 ± 0.0438 |
| 85 | hetero_gatv2 | `hetero_gatv2__h64__L3__d0.1__lr0.001__hd4__shared` | 3 | 0.5881 ± 0.0105 | 0.7520 ± 0.0103 | 0.4554 ± 0.0332 |
| 86 | hetero_sage_matched | `hetero_sage_matched__h64__L4__d0.25__lr0.0003__shared` | 3 | 0.5802 ± 0.0317 | 0.7543 ± 0.0249 | 0.4699 ± 0.0441 |
| 87 | hetero_gatv2 | `hetero_gatv2__h64__L4__d0.1__lr0.0003__hd4__shared` | 3 | 0.5796 ± 0.0158 | 0.7396 ± 0.0064 | 0.4563 ± 0.0427 |
| 88 | hetero_sage_matched | `hetero_sage_matched__h32__L4__d0.25__lr0.0003__shared` | 3 | 0.5779 ± 0.0123 | 0.7559 ± 0.0085 | 0.4751 ± 0.0497 |
| 89 | hetero_gatv2 | `hetero_gatv2__h64__L4__d0.1__lr0.001__hd4__shared` | 3 | 0.5770 ± 0.0180 | 0.7386 ± 0.0098 | 0.4685 ± 0.0034 |
| 90 | hetero_sage_matched | `hetero_sage_matched__h32__L3__d0.1__lr0.0003__shared` | 3 | 0.5764 ± 0.0158 | 0.7431 ± 0.0326 | 0.4863 ± 0.0093 |
| 91 | hetero_gatv2 | `hetero_gatv2__h32__L4__d0.1__lr0.001__hd4__shared` | 3 | 0.5702 ± 0.0180 | 0.7279 ± 0.0119 | 0.4634 ± 0.0466 |
| 92 | hetero_gatv2 | `hetero_gatv2__h32__L4__d0.1__lr0.0003__hd4__shared` | 3 | 0.5688 ± 0.0391 | 0.7433 ± 0.0332 | 0.4728 ± 0.0286 |
| 93 | hetero_sage_matched | `hetero_sage_matched__h64__L4__d0.1__lr0.001__shared` | 3 | 0.5682 ± 0.0199 | 0.7600 ± 0.0133 | 0.4797 ± 0.0481 |
| 94 | hetero_sage_matched | `hetero_sage_matched__h32__L4__d0.1__lr0.001__shared` | 3 | 0.5676 ± 0.0241 | 0.7443 ± 0.0199 | 0.4965 ± 0.0032 |
| 95 | hetero_gatv2 | `hetero_gatv2__h32__L2__d0.1__lr0.001__hd4__shared` | 3 | 0.5656 ± 0.1013 | 0.7443 ± 0.0485 | 0.4614 ± 0.1045 |
| 96 | hetero_gatv2 | `hetero_gatv2__h32__L4__d0.25__lr0.0003__hd4__shared` | 3 | 0.5602 ± 0.0355 | 0.7360 ± 0.0245 | 0.4574 ± 0.0621 |

## Top-2 per backbone (selection shortlist)

### hetero_gatv2

| rank | config_id | mean valid AUPRC ± std | n_seeds |
| ---: | --- | ---: | ---: |
| 1 | `hetero_gatv2__h32__L2__d0.25__lr0.0003__hd4__shared` | 0.6497 ± 0.0057 | 3 |
| 2 | `hetero_gatv2__h64__L2__d0.25__lr0.001__hd4__shared` | 0.6393 ± 0.0157 | 3 |

### hetero_sage_matched

| rank | config_id | mean valid AUPRC ± std | n_seeds |
| ---: | --- | ---: | ---: |
| 1 | `hetero_sage_matched__h32__L2__d0.25__lr0.0003__shared` | 0.6705 ± 0.0160 | 3 |
| 2 | `hetero_sage_matched__h32__L2__d0.25__lr0.001__shared` | 0.6485 ± 0.0568 | 3 |

### hgt

| rank | config_id | mean valid AUPRC ± std | n_seeds |
| ---: | --- | ---: | ---: |
| 1 | `hgt__h32__L2__d0.25__lr0.001__hd4__shared` | 0.7288 ± 0.0091 | 3 |
| 2 | `hgt__h32__L4__d0.25__lr0.001__hd4__shared` | 0.7262 ± 0.0334 | 3 |

### rgcn_matched

| rank | config_id | mean valid AUPRC ± std | n_seeds |
| ---: | --- | ---: | ---: |
| 1 | `rgcn_matched__h64__L3__d0.25__lr0.0003__b8__shared` | 0.6972 ± 0.0098 | 3 |
| 2 | `rgcn_matched__h32__L2__d0.25__lr0.0003__b8__shared` | 0.6933 ± 0.0185 | 3 |

## Suggested freeze candidate (clean screen only)

- Best mean valid AUPRC: `hgt__h32__L2__d0.25__lr0.001__hd4__shared` (0.7288 ± 0.0091, n=3)
- Protocol next: Top-2 / backbone → 6 scenarios × 3 seeds, then freeze one / backbone
- **Do not** use sealed test for this choice

