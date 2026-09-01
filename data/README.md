# Data

This project uses the Wisconsin Local Government Property Insurance Fund case-study data from the open-access *Loss Data Analytics* textbook.

The analysis requires two files:

- `CLAIMLEVEL.csv` – individual historical claim records
- `PropertyFundInsample.csv` – policy-year exposure and claim-frequency information

The files cover 2006–2010. The raw data is not stored in this repository; download it from the [Loss Data Analytics case study](https://openacttexts.github.io/Loss-Data-Analytics/ChapIntro.html) and place both CSVs in this folder before running the analysis.

One important feature is that 2,810 of the 6,258 claim records are below their listed policy deductible. I therefore treat `Claim` as the recorded ground-up or informational loss field supplied by the source rather than subtracting `Deduct` again. This interpretation is stated explicitly because it materially affects frequency modelling, although it has little direct effect on the selected XoL layers because their attachments begin at $50,000.
