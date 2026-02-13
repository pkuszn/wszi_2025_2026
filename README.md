### Import pliku .dmp w postgres
```bash
pg_restore -U chembl -d chembl_36 -j 4 --no-owner --no-privileges /backup/chembl_36_postgresql.dmp
```

### Notebooki
1. notebooks/notebook-1.ipynb - zapoznanie się z danymi, analiza pierwotnie na bazie sqlite. 
2. notebooks/notebook-2.ipynb i utils/notebook-3.ipynb - przeniesienie bazy do postgres, próba uruchomienia spark
3. <b>notebooks/notebook-4.ipynb - podejście do EDA</b>

### Widoki zmaterializowane
1. db/create-materialized-view - potrzebne zapytania tworzące widoki zmaterializowane dla aktywności, jednostek, molekuł etc

### parquets
1. notebooks/parquets/df_ml_with_scaffold - zapis danych po konwersji, usunięciu danych strukturalnych, usunięciu braków etc

