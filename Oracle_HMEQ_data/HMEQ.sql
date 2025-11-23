create table HMEQ_DATA
(
  bad     NUMBER(1),
  loan    NUMBER(15),
  mortdue NUMBER(15),
  value   NUMBER(15),
  reason  VARCHAR2(100),
  job     VARCHAR2(100),
  yoj     NUMBER,
  derog   NUMBER(2),
  delinq  NUMBER(2),
  clage   NUMBER,
  ninq    NUMBER(2),
  clno    NUMBER(2),
  debtinc NUMBER
)
/
-- Add comments to the table 
comment on table HMEQ_DATA
  is 'Predict clients who default on their loan(https://www.kaggle.com/datasets/ajay1735/hmeq-data)';
-- Add comments to the columns 
comment on column HMEQ_DATA.bad
  is '1 = client defaulted on loan 0 = loan repaid';
comment on column HMEQ_DATA.loan
  is 'Amount of the loan request';
comment on column HMEQ_DATA.mortdue
  is 'Amount due on existing mortgage';
comment on column HMEQ_DATA.value
  is 'Value of current property';
comment on column HMEQ_DATA.reason
  is 'DebtCon = debt consolidation HomeImp = home improvement';
comment on column HMEQ_DATA.job
  is 'Six occupational categories';
comment on column HMEQ_DATA.yoj
  is 'Years at present job';
comment on column HMEQ_DATA.derog
  is 'Number of major derogatory reports';
comment on column HMEQ_DATA.delinq
  is 'Number of delinquent credit lines';
comment on column HMEQ_DATA.clage
  is 'Age of oldest trade line in months';
comment on column HMEQ_DATA.ninq
  is 'Number of recent credit lines';
comment on column HMEQ_DATA.clno
  is 'Number of credit lines';
comment on column HMEQ_DATA.debtinc
  is 'Debt-to-income ratio';
