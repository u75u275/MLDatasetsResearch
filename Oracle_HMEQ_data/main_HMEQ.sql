DELETE FROM HMEQ_SETTINGS;

INSERT INTO HMEQ_SETTINGS VALUES ('ALGO_NAME', 'ALGO_DECISION_TREE');
INSERT INTO HMEQ_SETTINGS VALUES ('TREE_TERM_MAX_DEPTH', '10');        -- максимальна глибина дерева
INSERT INTO HMEQ_SETTINGS VALUES ('TREE_TERM_MINREC_NODE', '20');     -- мінімальна кількість записів для вузла
-- додайте інші доступні параметри Decision Tree за потребою
COMMIT;


BEGIN
  DBMS_DATA_MINING.DROP_MODEL('HMEQ_LOAN_DEFAULT_MODEL');
EXCEPTION WHEN OTHERS THEN NULL; END;
/

BEGIN
  DBMS_DATA_MINING.CREATE_MODEL(
    model_name          => 'HMEQ_LOAN_DEFAULT_MODEL',
    mining_function     => DBMS_DATA_MINING.CLASSIFICATION,
    data_table_name     => 'HMEQ_DATA',
    case_id_column_name => 'loan',
    target_column_name  => 'bad',
    settings_table_name => 'HMEQ_SETTINGS');
END;
/




-- Далі рахуйте точність, recall, precision та плутанину через групування.
SELECT
    actual,
    CASE WHEN probability > 0.3 THEN 1 ELSE 0 END AS predicted,
    COUNT(*)
FROM (
    SELECT bad AS actual,
           PREDICTION_PROBABILITY(HMEQ_LOAN_DEFAULT_MODEL USING *) AS probability
    FROM HMEQ_DATA
)
GROUP BY actual, CASE WHEN probability > 0.3 THEN 1 ELSE 0 END;
