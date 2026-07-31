@echo off

echo ============================================
echo Running PRSice...
echo ============================================

"C:\Program Files\R\R-4.6.1\bin\Rscript.exe" "C:\MammoRiskPlus\tools\prsice\PRSice.R" --prsice "C:\MammoRiskPlus\tools\prsice\PRSice_win64.exe" --base "C:\MammoRiskPlus\results\prsice_base.txt" --target "C:\MammoRiskPlus\data\genotypes\chr1_phase3" --beta --stat BETA --a1 A1 --a2 A2 --chr CHR --bp BP --snp SNP --pvalue P --no-clump --no-regress --score sum --out "C:\MammoRiskPlus\results\prs_output"

pause