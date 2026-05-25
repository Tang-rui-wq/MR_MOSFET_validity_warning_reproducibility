# Push instructions

This local repository has already been initialized and committed.

Current local path:

```text
C:\Users\Tangrui\Desktop\MR_MOSFET_validity_warning_reproducibility_20260525
```

The repository has been published at:

```text
https://github.com/Tang-rui-wq/MR_MOSFET_validity_warning_reproducibility
```

## Option A: create a new public GitHub repository

```bash
gh auth login
cd C:\Users\Tangrui\Desktop\MR_MOSFET_validity_warning_reproducibility_20260525
gh repo create MR_MOSFET_validity_warning_reproducibility --public --source . --remote origin --push
```

## Option B: push to an existing GitHub repository

Replace `OWNER/REPO` with the real repository path.

```bash
gh auth login
cd C:\Users\Tangrui\Desktop\MR_MOSFET_validity_warning_reproducibility_20260525
git remote add origin https://github.com/OWNER/REPO.git
git push -u origin main
```

After pushing, archive the public GitHub repository on Zenodo and add the Zenodo DOI to `README.md`, `DATA_AVAILABILITY.md`, and the manuscript Data Availability statement.
