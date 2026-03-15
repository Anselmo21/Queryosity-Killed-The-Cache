-- Secondary indexes for TPC-H foreign key columns.
-- PostgreSQL does not auto-create indexes on FK columns.
-- These indexes are required for reasonable query performance,
-- especially at scale factors >= 10.

-- lineitem: most queries join on these columns
CREATE INDEX IF NOT EXISTS lineitem_partkey_idx ON lineitem (l_partkey);
CREATE INDEX IF NOT EXISTS lineitem_suppkey_idx ON lineitem (l_suppkey);
CREATE INDEX IF NOT EXISTS lineitem_partkey_suppkey_idx ON lineitem (l_partkey, l_suppkey);
CREATE INDEX IF NOT EXISTS lineitem_shipdate_idx ON lineitem (l_shipdate);

-- orders: joined via o_custkey
CREATE INDEX IF NOT EXISTS orders_custkey_idx ON orders (o_custkey);

-- partsupp: FK columns
CREATE INDEX IF NOT EXISTS partsupp_partkey_idx ON partsupp (ps_partkey);
CREATE INDEX IF NOT EXISTS partsupp_suppkey_idx ON partsupp (ps_suppkey);

-- supplier: FK column
CREATE INDEX IF NOT EXISTS supplier_nationkey_idx ON supplier (s_nationkey);

-- customer: FK column
CREATE INDEX IF NOT EXISTS customer_nationkey_idx ON customer (c_nationkey);

-- nation: FK column
CREATE INDEX IF NOT EXISTS nation_regionkey_idx ON nation (n_regionkey);
