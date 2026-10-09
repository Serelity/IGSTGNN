# Chattanooga 数据问题询问草稿（未发送）

用途：取得能够核实数值处理链的最小材料。没有指定或验证收件邮箱，没有发送邮件、提交 issue 或数据申请。

Subject: Question about aggregate volume and occupancy values in the Chattanooga dataset

Dear Dr. Berres and Dr. Moriano,

We are studying incident-aware traffic forecasting with traffic-flow constraints and have been examining your Chattanooga dataset (Zenodo DOI: 10.5281/zenodo.7964288) and its associated Data in Brief and Expert Systems with Applications papers.

The data paper describes summing vehicle counts across lanes and averaging speed and occupancy. In the published CSV files, we found some very large numeric values whose interpretation we would like to clarify before using the data for physical modeling.

For example, in the bestData injury/fatality candidate subset for I-75 and I-24, we found the literal volume value `21483742148374` and occupancy value `1.0224886583081654e+22` at the same observation, two minutes before the report time, in the `(i-2)` columns. These strings occur in the downloaded files themselves. Across this count-matched 55-event subset, 17 windows have occupancy above 100 within the published −4 to +7 minute feature boundaries when considering all 11 sensors.

We have not verified that this candidate subset or these published values are identical to your final model inputs, and we do not infer the cause of these values or draw conclusions about the detection results.

Could you please help clarify:

1. Whether these are valid encoded values, known aggregation/export issues, or values excluded or transformed by the released experimental code, and whether a corrected release exists.
2. Whether the lane-to-station aggregation script and a small corresponding lane-level example could be shared. A single abnormal station/time example would already help; we do not need the full raw archive.
3. Whether the supplementary Code Ocean capsule (DOI: 10.24433/CO.8627573.v1) includes this conversion step, or whether there is another public location for that code.

For file identity, our MD5 checks match the Zenodo record: annotatedData.zip `9ba05b3ad4fe358998f6c17fd882a79a`; metaData.zip `394d6c5615fc96c5f2ef979d2cf7275d`.

Thank you for making the dataset and methods available.

Best regards,
[Researcher's name and affiliation]

---

草稿中 17/55 仅指此次候选子集和时间边界，与之前整个半小时窗口的 18/55 不冲突。发送前补真实署名、验证作者当前联系地址；不使用未经验证的邮箱。
