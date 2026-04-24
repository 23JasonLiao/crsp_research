SELECT 
    h.crsp_fundno,
    h.crsp_portno,
    h.crsp_cl_grp,
    h.fund_name,
    h.nasdaq AS ticker,
    h.ncusip,
    h.mgmt_name,
    h.mgmt_cd,
    h.mgr_name,
    h.mgr_dt,
    h.adv_name,
    h.open_to_inv,
    h.retail_fund,
    h.inst_fund,
    h.m_fund,
    h.index_fund_flag,
    h.vau_fund,
    h.et_flag,
    h.first_offer_dt,
    h.dead_flag,
    h.delist_cd,
    h.merge_fundno,
    mret.caldt,
    EXTRACT(YEAR FROM mret.caldt) AS caldt_year,
    EXTRACT(MONTH FROM mret.caldt) AS caldt_month,
    (EXTRACT(YEAR FROM mret.caldt) * 100 + EXTRACT(MONTH FROM mret.caldt)) AS caldt_time,
    EXTRACT(YEAR FROM h.first_offer_dt) AS first_caldt_year,
    EXTRACT(MONTH FROM h.first_offer_dt) AS first_caldt_month,
    (EXTRACT(YEAR FROM h.first_offer_dt) * 100 + EXTRACT(MONTH FROM h.first_offer_dt)) AS first_caldt_time,
    (mret.caldt - h.first_offer_dt) / 365.25 AS age_tmp,
    EXTRACT(YEAR FROM AGE(mret.caldt, h.first_offer_dt)) AS age,
    mnav.mnav,
    mret.mret,
    mtna.mtna,
    fees.actual_12b1,
    fees.max_12b1,
    fees.exp_ratio,
    fees.mgmt_fee,
    fees.turn_ratio,
    fees.fiscal_yearend,
    sty.crsp_obj_cd,
    sty.si_obj_cd,
    sty.accrual_fund,
    sty.sales_restrict,
    sty.wbrger_obj_cd,
    sty.policy,
    sty.lipper_class,
    sty.lipper_class_name,
    sty.lipper_obj_cd,
    sty.lipper_obj_name,
    sty.lipper_tax_cd,
    sty.lipper_asset_cd,
    ff.crsp_portno_check,
    ff.flow_report_dt,
    ff.trans_dt,
    ff.new_sls,
    ff.rein_sls,
    ff.oth_sls,
    ff.redemp,
    div.dis_type,
    div.dis_amt,
    div.reinvest_nav,
    div.spl_ratio
FROM MONTHLY_RETURNS mret
LEFT JOIN MONTHLY_NAV mnav ON mret.crsp_fundno = mnav.crsp_fundno AND mret.caldt = mnav.caldt
LEFT JOIN MONTHLY_TNA mtna ON mret.crsp_fundno = mtna.crsp_fundno AND mret.caldt = mtna.caldt
LEFT JOIN FUND_HDR h ON mret.crsp_fundno = h.crsp_fundno
LEFT JOIN FUND_FEES fees ON mret.crsp_fundno = fees.crsp_fundno 
    AND mret.caldt >= fees.begdt AND (mret.caldt <= fees.enddt OR fees.enddt IS NULL)
LEFT JOIN FUND_STYLE sty ON mret.crsp_fundno = sty.crsp_fundno 
    AND mret.caldt >= sty.begdt AND (mret.caldt <= sty.enddt OR sty.enddt IS NULL)
LEFT JOIN FUND_FLOWS ff ON h.crsp_portno = ff.crsp_portno_check 
    AND EXTRACT(YEAR FROM mret.caldt) = EXTRACT(YEAR FROM ff.trans_dt)
    AND EXTRACT(MONTH FROM mret.caldt) = EXTRACT(MONTH FROM ff.trans_dt)
LEFT JOIN DIVIDENDS div ON mret.crsp_fundno = div.crsp_fundno AND mret.caldt = div.caldt

-- 修正後的篩選條件：只要 Balanced 且年份在 2010 以前
WHERE (TRIM(sty.lipper_obj_cd) = 'B' OR TRIM(sty.lipper_obj_name) = 'Balanced')
  AND EXTRACT(YEAR FROM mret.caldt) < 2010;