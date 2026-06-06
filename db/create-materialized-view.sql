---Tworzy widok zmaterializowany, wyrzucający wszystkie jednostki w IC50.
---Jakie jednostki dominują w danych?
---Które można łatwo przekonwertować?
CREATE MATERIALIZED VIEW chembl_ic50_units_human AS
SELECT
    a.standard_units,
    COUNT(*) AS n_records,
    MIN(a.standard_value) AS min_value,
    MAX(a.standard_value) AS max_value,
    AVG(a.standard_value) AS mean_value,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY a.standard_value) AS median_value
FROM public.activities a
JOIN public.assays asy
    ON a.assay_id = asy.assay_id
WHERE
    a.standard_type = 'IC50'
    AND a.relation = '='               -- tylko dokładne wartości
    AND a.standard_value IS NOT NULL
    AND a.standard_value > 0
    AND asy.assay_organism = 'Homo sapiens' -- tylko assay ludzkie
GROUP BY a.standard_units
ORDER BY n_records DESC;

--- Tworzy widok zmaterializowany, który łączy dwie tabele - `activities` i `compound_structures`. Pierwsza tabela zawiera wyniki eksperymentów (IC50 values, compounds).
--- Druga tabela zawiera strukturę SMILES każdej molekuły. Dane są wyfiltrowane po IC50 w nM, tylko pozytywne wartości.
--- Otrzymane wartości logarytmujemy na potrzeby łatwiejszego modelowania.
--- Pod logarytm podstawiamy wartośc IC50 w molach.
--- Związki tylko testowane na ludziach.
CREATE MATERIALIZED VIEW chembl_pic50_compounds AS WITH filtered AS (
    SELECT
        a.activity_id,
        a.assay_id,
        a.molregno,
        cs.canonical_smiles,
        a.standard_value,
        a.standard_units,
        a.standard_type,
        a.relation,
        a.pchembl_value,
        a.data_validity_comment,
        a.activity_comment,
        a.src_id,

        -- pIC50 computation (IC50 in nM)
        -LOG(10, a.standard_value * 1e-9) AS pIC50

    FROM public.activities a
    JOIN public.compound_structures cs
      ON a.molregno = cs.molregno
    JOIN public.assays asy
      ON a.assay_id = asy.assay_id

    WHERE
        a.standard_type = 'IC50' -- tylko wpisy z IC50
        AND a.standard_units = 'nM'
        AND a.relation = '='
        AND a.standard_value IS NOT NULL
        AND a.standard_value > 0
        AND a.data_validity_comment IS NULL
        AND asy.assay_organism = 'Homo sapiens'
)

SELECT
    canonical_smiles,
    COUNT(*)               AS "n_assays",
    AVG(pIC50)              AS "pIC50_mean",
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pIC50) AS "pIC50_median",
    MIN(pIC50)              AS "pIC50_min",
    MAX(pIC50)              AS "pIC50_max"
FROM filtered
GROUP BY canonical_smiles;



--- Tworzy widok zmaterializowany, który łączy dwie tabele - `activities` i `compound_structures`. Pierwsza tabela zawiera wyniki eksperymentów (IC50 values, compounds).
--- Druga tabela zawiera strukturę SMILES każdej molekuły. Dane są wyfiltrowane po IC50 w nM, tylko pozytywne wartości.
--- Otrzymane wartości logarytmujemy na potrzeby łatwiejszego modelowania.
--- Pod logarytm podstawiamy wartośc IC50 w molach.
--- Związki tylko testowane na ludziach.
--- Dodatkowo, przelicza konwertuje do bazowej jednostki.
CREATE MATERIALIZED VIEW chembl_pic50_compounds_all_units AS
WITH filtered AS (
    SELECT
        a.activity_id,
        a.assay_id,
        a.molregno,
        cs.canonical_smiles,
        a.standard_value,
        a.standard_units,
        a.standard_type,
        a.relation,
        a.pchembl_value,
        a.data_validity_comment,
        a.activity_comment,
        a.src_id,

        CASE
            WHEN a.standard_units = 'nM' THEN a.standard_value * 1e-9
            WHEN a.standard_units = 'uM' THEN a.standard_value * 1e-6
            WHEN a.standard_units = 'mM' THEN a.standard_value * 1e-3
            WHEN a.standard_units = 'pM' THEN a.standard_value * 1e-12
            ELSE NULL  -- jednostki niestandardowe ignorujemy w logarytmie
        END AS ic50_molar,

        CASE
            WHEN a.standard_units IN ('nM','uM','mM','pM') THEN -LOG(10, a.standard_value *
                 CASE
                     WHEN a.standard_units = 'nM' THEN 1e-9
                     WHEN a.standard_units = 'uM' THEN 1e-6
                     WHEN a.standard_units = 'mM' THEN 1e-3
                     WHEN a.standard_units = 'pM' THEN 1e-12
                 END)
            ELSE NULL
        END AS pIC50

    FROM public.activities a
    JOIN public.compound_structures cs
      ON a.molregno = cs.molregno
    JOIN public.assays asy
      ON a.assay_id = asy.assay_id
    WHERE
        a.standard_type = 'IC50'
        AND a.relation = '='
        AND a.standard_value IS NOT NULL
        AND a.standard_value > 0
        AND asy.assay_organism = 'Homo sapiens'
        AND a.data_validity_comment IS NULL
)

SELECT
    canonical_smiles,
    COUNT(*) AS n_assays,
    AVG(pIC50) AS pIC50_mean,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pIC50) AS pIC50_median,
    MIN(pIC50) AS pIC50_min,
    MAX(pIC50) AS pIC50_max,
    ARRAY_AGG(DISTINCT standard_units) AS all_units  -- pokazuje wszystkie jednostki występujące dla danego SMILES
FROM filtered
WHERE pIC50 IS NOT NULL  -- tylko jednostki, które udało się przeliczyć
GROUP BY canonical_smiles
ORDER BY n_assays DESC;

--- Widok zmaterializowany do pobierania deskryptorów molekularnych w CHemBL
CREATE MATERIALIZED VIEW chembl_molecular_descriptors AS
SELECT
    md.molregno,
    md.pref_name,
    cs.canonical_smiles,
    cp.full_mwt,             -- masa molowa
    cp.alogp,                -- logP
    cp.psa,                  -- polar surface area
    cp.hba,                  -- liczba akceptorów H
    cp.hbd,                  -- liczba donorów H
    cp.ro5_violations,       -- naruszenia reguły 5 Lipinskiego
    cp.qed_weighted           -- QED (drug-likeness)
FROM public.molecule_dictionary md
JOIN public.compound_properties cp
  ON md.molregno = cp.molregno
JOIN public.compound_structures cs
  ON md.molregno = cs.molregno
ORDER BY md.molregno;

---IC50 po konwersji do bazowej jednostki, bez agregacji
CREATE MATERIALIZED VIEW chembl_ic50_all_converted AS
SELECT
    a.activity_id,
    a.assay_id,
    a.molregno,
    cs.canonical_smiles,
    a.standard_value,
    a.standard_units,
    a.standard_type,
    a.relation,
    a.pchembl_value,
    a.data_validity_comment,
    a.activity_comment,
    a.src_id,

    -- przeliczenie do moli
    CASE
        WHEN a.standard_units = 'nM' THEN a.standard_value * 1e-9
        WHEN a.standard_units = 'uM' THEN a.standard_value * 1e-6
        WHEN a.standard_units = 'mM' THEN a.standard_value * 1e-3
        WHEN a.standard_units = 'pM' THEN a.standard_value * 1e-12
        ELSE NULL
    END AS ic50_molar,

    -- obliczenie pIC50
    CASE
        WHEN a.standard_units IN ('nM','uM','mM','pM') THEN 
            -LOG(10, a.standard_value *
                 CASE
                     WHEN a.standard_units = 'nM' THEN 1e-9
                     WHEN a.standard_units = 'uM' THEN 1e-6
                     WHEN a.standard_units = 'mM' THEN 1e-3
                     WHEN a.standard_units = 'pM' THEN 1e-12
                 END)
        ELSE NULL
    END AS pIC50

FROM public.activities a
JOIN public.compound_structures cs
  ON a.molregno = cs.molregno
JOIN public.assays asy
  ON a.assay_id = asy.assay_id
WHERE a.standard_type = 'IC50'
  AND a.relation = '='
  AND a.standard_value IS NOT NULL
  AND a.standard_value > 0
  AND asy.assay_organism = 'Homo sapiens'
  AND a.data_validity_comment IS NULL;

--- Widok zmaterializowany do pobierania deskryptorów
CREATE MATERIALIZED VIEW chembl_pic50_ml_ready AS
WITH filtered AS (
    SELECT
        a.activity_id,
        a.assay_id,
        a.molregno,
        cs.canonical_smiles,
        a.standard_value,
        a.standard_units,
        a.standard_type,
        a.relation,
        a.pchembl_value,
        a.data_validity_comment,
        a.activity_comment,
        a.src_id,

        -- IC50 -> molar conversion
        CASE
            WHEN a.standard_units = 'nM' THEN a.standard_value * 1e-9
            WHEN a.standard_units = 'uM' THEN a.standard_value * 1e-6
            WHEN a.standard_units = 'mM' THEN a.standard_value * 1e-3
            WHEN a.standard_units = 'pM' THEN a.standard_value * 1e-12
            ELSE NULL
        END AS ic50_molar,

        -- pIC50
        CASE
            WHEN a.standard_units IN ('nM','uM','mM','pM')
                 AND a.standard_value > 0
            THEN -LOG(10, a.standard_value *
                CASE
                    WHEN a.standard_units = 'nM' THEN 1e-9
                    WHEN a.standard_units = 'uM' THEN 1e-6
                    WHEN a.standard_units = 'mM' THEN 1e-3
                    WHEN a.standard_units = 'pM' THEN 1e-12
                END)
            ELSE NULL
        END AS pIC50

    FROM public.activities a
    JOIN public.compound_structures cs
        ON a.molregno = cs.molregno
    JOIN public.assays asy
        ON a.assay_id = asy.assay_id
    WHERE
        a.standard_type = 'IC50'
        AND a.relation = '='
        AND a.standard_value IS NOT NULL
        AND a.standard_value > 0
        AND a.data_validity_comment IS NULL
        AND asy.assay_organism = 'Homo sapiens'
)

SELECT
    canonical_smiles,
    assay_id,
    molregno,
    pIC50,
    standard_units,
    ic50_molar
FROM filtered
WHERE pIC50 IS NOT NULL;