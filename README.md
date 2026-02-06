### Import pliku .dmp w postgres
```bash
pg_restore -U chembl -d chembl_36 -j 4 --no-owner --no-privileges /backup/chembl_36_postgresql.dmp
```

### Notebooki
1. utils/notebook-1.ipynb - zapoznanie się z danymi, analiza pierwotnie na bazie sqlite. 
2. utils/notebook-2.ipynb i utils/notebook-3.ipynb - przeniesienie bazy do postgres, próba uruchomienia spark
3. utils/notebook-4.ipynb - podejście do EDA

