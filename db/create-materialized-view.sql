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