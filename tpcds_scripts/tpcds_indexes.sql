-- Secondary indexes for TPC-DS foreign key columns.
-- PostgreSQL does not auto-create indexes on FK columns.
-- These indexes are required for reasonable query performance,
-- especially at scale factors >= 10.

-- call_center
CREATE INDEX IF NOT EXISTS call_center_cc_closed_date_sk_idx ON call_center (cc_closed_date_sk);
CREATE INDEX IF NOT EXISTS call_center_cc_open_date_sk_idx ON call_center (cc_open_date_sk);

-- catalog_page
CREATE INDEX IF NOT EXISTS catalog_page_cp_end_date_sk_idx ON catalog_page (cp_end_date_sk);
CREATE INDEX IF NOT EXISTS catalog_page_cp_start_date_sk_idx ON catalog_page (cp_start_date_sk);

-- catalog_returns
CREATE INDEX IF NOT EXISTS catalog_returns_cr_call_center_sk_idx ON catalog_returns (cr_call_center_sk);
CREATE INDEX IF NOT EXISTS catalog_returns_cr_catalog_page_sk_idx ON catalog_returns (cr_catalog_page_sk);
CREATE INDEX IF NOT EXISTS catalog_returns_cr_item_sk_idx ON catalog_returns (cr_item_sk);
CREATE INDEX IF NOT EXISTS catalog_returns_cr_reason_sk_idx ON catalog_returns (cr_reason_sk);
CREATE INDEX IF NOT EXISTS catalog_returns_cr_refunded_addr_sk_idx ON catalog_returns (cr_refunded_addr_sk);
CREATE INDEX IF NOT EXISTS catalog_returns_cr_refunded_cdemo_sk_idx ON catalog_returns (cr_refunded_cdemo_sk);
CREATE INDEX IF NOT EXISTS catalog_returns_cr_refunded_customer_sk_idx ON catalog_returns (cr_refunded_customer_sk);
CREATE INDEX IF NOT EXISTS catalog_returns_cr_refunded_hdemo_sk_idx ON catalog_returns (cr_refunded_hdemo_sk);
CREATE INDEX IF NOT EXISTS catalog_returns_cr_returned_date_sk_idx ON catalog_returns (cr_returned_date_sk);
CREATE INDEX IF NOT EXISTS catalog_returns_cr_returned_time_sk_idx ON catalog_returns (cr_returned_time_sk);
CREATE INDEX IF NOT EXISTS catalog_returns_cr_returning_addr_sk_idx ON catalog_returns (cr_returning_addr_sk);
CREATE INDEX IF NOT EXISTS catalog_returns_cr_returning_cdemo_sk_idx ON catalog_returns (cr_returning_cdemo_sk);
CREATE INDEX IF NOT EXISTS catalog_returns_cr_returning_customer_sk_idx ON catalog_returns (cr_returning_customer_sk);
CREATE INDEX IF NOT EXISTS catalog_returns_cr_returning_hdemo_sk_idx ON catalog_returns (cr_returning_hdemo_sk);
CREATE INDEX IF NOT EXISTS catalog_returns_cr_ship_mode_sk_idx ON catalog_returns (cr_ship_mode_sk);
CREATE INDEX IF NOT EXISTS catalog_returns_cr_warehouse_sk_idx ON catalog_returns (cr_warehouse_sk);

-- catalog_sales
CREATE INDEX IF NOT EXISTS catalog_sales_cs_bill_addr_sk_idx ON catalog_sales (cs_bill_addr_sk);
CREATE INDEX IF NOT EXISTS catalog_sales_cs_bill_cdemo_sk_idx ON catalog_sales (cs_bill_cdemo_sk);
CREATE INDEX IF NOT EXISTS catalog_sales_cs_bill_customer_sk_idx ON catalog_sales (cs_bill_customer_sk);
CREATE INDEX IF NOT EXISTS catalog_sales_cs_bill_hdemo_sk_idx ON catalog_sales (cs_bill_hdemo_sk);
CREATE INDEX IF NOT EXISTS catalog_sales_cs_call_center_sk_idx ON catalog_sales (cs_call_center_sk);
CREATE INDEX IF NOT EXISTS catalog_sales_cs_catalog_page_sk_idx ON catalog_sales (cs_catalog_page_sk);
CREATE INDEX IF NOT EXISTS catalog_sales_cs_item_sk_idx ON catalog_sales (cs_item_sk);
CREATE INDEX IF NOT EXISTS catalog_sales_cs_promo_sk_idx ON catalog_sales (cs_promo_sk);
CREATE INDEX IF NOT EXISTS catalog_sales_cs_ship_addr_sk_idx ON catalog_sales (cs_ship_addr_sk);
CREATE INDEX IF NOT EXISTS catalog_sales_cs_ship_cdemo_sk_idx ON catalog_sales (cs_ship_cdemo_sk);
CREATE INDEX IF NOT EXISTS catalog_sales_cs_ship_customer_sk_idx ON catalog_sales (cs_ship_customer_sk);
CREATE INDEX IF NOT EXISTS catalog_sales_cs_ship_date_sk_idx ON catalog_sales (cs_ship_date_sk);
CREATE INDEX IF NOT EXISTS catalog_sales_cs_ship_hdemo_sk_idx ON catalog_sales (cs_ship_hdemo_sk);
CREATE INDEX IF NOT EXISTS catalog_sales_cs_ship_mode_sk_idx ON catalog_sales (cs_ship_mode_sk);
CREATE INDEX IF NOT EXISTS catalog_sales_cs_sold_date_sk_idx ON catalog_sales (cs_sold_date_sk);
CREATE INDEX IF NOT EXISTS catalog_sales_cs_sold_time_sk_idx ON catalog_sales (cs_sold_time_sk);
CREATE INDEX IF NOT EXISTS catalog_sales_cs_warehouse_sk_idx ON catalog_sales (cs_warehouse_sk);

-- customer
CREATE INDEX IF NOT EXISTS customer_c_current_addr_sk_idx ON customer (c_current_addr_sk);
CREATE INDEX IF NOT EXISTS customer_c_current_cdemo_sk_idx ON customer (c_current_cdemo_sk);
CREATE INDEX IF NOT EXISTS customer_c_current_hdemo_sk_idx ON customer (c_current_hdemo_sk);
CREATE INDEX IF NOT EXISTS customer_c_first_sales_date_sk_idx ON customer (c_first_sales_date_sk);
CREATE INDEX IF NOT EXISTS customer_c_first_shipto_date_sk_idx ON customer (c_first_shipto_date_sk);

-- household_demographics
CREATE INDEX IF NOT EXISTS household_demographics_hd_income_band_sk_idx ON household_demographics (hd_income_band_sk);

-- inventory
CREATE INDEX IF NOT EXISTS inventory_inv_date_sk_idx ON inventory (inv_date_sk);
CREATE INDEX IF NOT EXISTS inventory_inv_item_sk_idx ON inventory (inv_item_sk);
CREATE INDEX IF NOT EXISTS inventory_inv_warehouse_sk_idx ON inventory (inv_warehouse_sk);

-- promotion
CREATE INDEX IF NOT EXISTS promotion_p_end_date_sk_idx ON promotion (p_end_date_sk);
CREATE INDEX IF NOT EXISTS promotion_p_item_sk_idx ON promotion (p_item_sk);
CREATE INDEX IF NOT EXISTS promotion_p_start_date_sk_idx ON promotion (p_start_date_sk);

-- store
CREATE INDEX IF NOT EXISTS store_s_closed_date_sk_idx ON store (s_closed_date_sk);

-- store_returns
CREATE INDEX IF NOT EXISTS store_returns_sr_addr_sk_idx ON store_returns (sr_addr_sk);
CREATE INDEX IF NOT EXISTS store_returns_sr_cdemo_sk_idx ON store_returns (sr_cdemo_sk);
CREATE INDEX IF NOT EXISTS store_returns_sr_customer_sk_idx ON store_returns (sr_customer_sk);
CREATE INDEX IF NOT EXISTS store_returns_sr_hdemo_sk_idx ON store_returns (sr_hdemo_sk);
CREATE INDEX IF NOT EXISTS store_returns_sr_item_sk_idx ON store_returns (sr_item_sk);
CREATE INDEX IF NOT EXISTS store_returns_sr_reason_sk_idx ON store_returns (sr_reason_sk);
CREATE INDEX IF NOT EXISTS store_returns_sr_return_time_sk_idx ON store_returns (sr_return_time_sk);
CREATE INDEX IF NOT EXISTS store_returns_sr_returned_date_sk_idx ON store_returns (sr_returned_date_sk);
CREATE INDEX IF NOT EXISTS store_returns_sr_store_sk_idx ON store_returns (sr_store_sk);

-- store_sales
CREATE INDEX IF NOT EXISTS store_sales_ss_addr_sk_idx ON store_sales (ss_addr_sk);
CREATE INDEX IF NOT EXISTS store_sales_ss_cdemo_sk_idx ON store_sales (ss_cdemo_sk);
CREATE INDEX IF NOT EXISTS store_sales_ss_customer_sk_idx ON store_sales (ss_customer_sk);
CREATE INDEX IF NOT EXISTS store_sales_ss_hdemo_sk_idx ON store_sales (ss_hdemo_sk);
CREATE INDEX IF NOT EXISTS store_sales_ss_item_sk_idx ON store_sales (ss_item_sk);
CREATE INDEX IF NOT EXISTS store_sales_ss_promo_sk_idx ON store_sales (ss_promo_sk);
CREATE INDEX IF NOT EXISTS store_sales_ss_sold_date_sk_idx ON store_sales (ss_sold_date_sk);
CREATE INDEX IF NOT EXISTS store_sales_ss_sold_time_sk_idx ON store_sales (ss_sold_time_sk);
CREATE INDEX IF NOT EXISTS store_sales_ss_store_sk_idx ON store_sales (ss_store_sk);

-- web_page
CREATE INDEX IF NOT EXISTS web_page_wp_access_date_sk_idx ON web_page (wp_access_date_sk);
CREATE INDEX IF NOT EXISTS web_page_wp_creation_date_sk_idx ON web_page (wp_creation_date_sk);

-- web_returns
CREATE INDEX IF NOT EXISTS web_returns_wr_item_sk_idx ON web_returns (wr_item_sk);
CREATE INDEX IF NOT EXISTS web_returns_wr_reason_sk_idx ON web_returns (wr_reason_sk);
CREATE INDEX IF NOT EXISTS web_returns_wr_refunded_addr_sk_idx ON web_returns (wr_refunded_addr_sk);
CREATE INDEX IF NOT EXISTS web_returns_wr_refunded_cdemo_sk_idx ON web_returns (wr_refunded_cdemo_sk);
CREATE INDEX IF NOT EXISTS web_returns_wr_refunded_customer_sk_idx ON web_returns (wr_refunded_customer_sk);
CREATE INDEX IF NOT EXISTS web_returns_wr_refunded_hdemo_sk_idx ON web_returns (wr_refunded_hdemo_sk);
CREATE INDEX IF NOT EXISTS web_returns_wr_returned_date_sk_idx ON web_returns (wr_returned_date_sk);
CREATE INDEX IF NOT EXISTS web_returns_wr_returned_time_sk_idx ON web_returns (wr_returned_time_sk);
CREATE INDEX IF NOT EXISTS web_returns_wr_returning_addr_sk_idx ON web_returns (wr_returning_addr_sk);
CREATE INDEX IF NOT EXISTS web_returns_wr_returning_cdemo_sk_idx ON web_returns (wr_returning_cdemo_sk);
CREATE INDEX IF NOT EXISTS web_returns_wr_returning_customer_sk_idx ON web_returns (wr_returning_customer_sk);
CREATE INDEX IF NOT EXISTS web_returns_wr_returning_hdemo_sk_idx ON web_returns (wr_returning_hdemo_sk);
CREATE INDEX IF NOT EXISTS web_returns_wr_web_page_sk_idx ON web_returns (wr_web_page_sk);

-- web_sales
CREATE INDEX IF NOT EXISTS web_sales_ws_bill_addr_sk_idx ON web_sales (ws_bill_addr_sk);
CREATE INDEX IF NOT EXISTS web_sales_ws_bill_cdemo_sk_idx ON web_sales (ws_bill_cdemo_sk);
CREATE INDEX IF NOT EXISTS web_sales_ws_bill_customer_sk_idx ON web_sales (ws_bill_customer_sk);
CREATE INDEX IF NOT EXISTS web_sales_ws_bill_hdemo_sk_idx ON web_sales (ws_bill_hdemo_sk);
CREATE INDEX IF NOT EXISTS web_sales_ws_item_sk_idx ON web_sales (ws_item_sk);
CREATE INDEX IF NOT EXISTS web_sales_ws_promo_sk_idx ON web_sales (ws_promo_sk);
CREATE INDEX IF NOT EXISTS web_sales_ws_ship_addr_sk_idx ON web_sales (ws_ship_addr_sk);
CREATE INDEX IF NOT EXISTS web_sales_ws_ship_cdemo_sk_idx ON web_sales (ws_ship_cdemo_sk);
CREATE INDEX IF NOT EXISTS web_sales_ws_ship_customer_sk_idx ON web_sales (ws_ship_customer_sk);
CREATE INDEX IF NOT EXISTS web_sales_ws_ship_date_sk_idx ON web_sales (ws_ship_date_sk);
CREATE INDEX IF NOT EXISTS web_sales_ws_ship_hdemo_sk_idx ON web_sales (ws_ship_hdemo_sk);
CREATE INDEX IF NOT EXISTS web_sales_ws_ship_mode_sk_idx ON web_sales (ws_ship_mode_sk);
CREATE INDEX IF NOT EXISTS web_sales_ws_sold_date_sk_idx ON web_sales (ws_sold_date_sk);
CREATE INDEX IF NOT EXISTS web_sales_ws_sold_time_sk_idx ON web_sales (ws_sold_time_sk);
CREATE INDEX IF NOT EXISTS web_sales_ws_warehouse_sk_idx ON web_sales (ws_warehouse_sk);
CREATE INDEX IF NOT EXISTS web_sales_ws_web_page_sk_idx ON web_sales (ws_web_page_sk);
CREATE INDEX IF NOT EXISTS web_sales_ws_web_site_sk_idx ON web_sales (ws_web_site_sk);

-- web_site
CREATE INDEX IF NOT EXISTS web_site_web_close_date_sk_idx ON web_site (web_close_date_sk);
CREATE INDEX IF NOT EXISTS web_site_web_open_date_sk_idx ON web_site (web_open_date_sk);
