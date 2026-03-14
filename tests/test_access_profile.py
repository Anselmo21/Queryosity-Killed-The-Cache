from src.scheduler.access_profile import AccessProfile, build_access_profile


class TestAccessProfile:
    def test_total_pages(self):
        p = AccessProfile(query_id="q1", table_pages={"A": 10, "B": 20})
        assert p.total_pages == 30

    def test_empty_profile(self):
        p = AccessProfile(query_id="q0", table_pages={})
        assert p.total_pages == 0


class TestBuildAccessProfile:
    def test_from_explain_without_analyze(self):
        """Simulate a plan tree without block statistics."""
        plan = {
            "Plan": {
                "Node Type": "Hash Join",
                "Plans": [
                    {
                        "Node Type": "Seq Scan",
                        "Relation Name": "lineitem",
                        "Plan Rows": 1000,
                        "Plan Width": 100,
                    },
                    {
                        "Node Type": "Hash",
                        "Plans": [
                            {
                                "Node Type": "Seq Scan",
                                "Relation Name": "orders",
                                "Plan Rows": 200,
                                "Plan Width": 50,
                            }
                        ],
                    },
                ],
            }
        }

        profile = build_access_profile("q1", plan)
        assert profile.query_id == "q1"
        assert "lineitem" in profile.table_pages
        assert "orders" in profile.table_pages
        # lineitem: ceil(1000 * 100 / 8192) = ceil(12.2) = 13
        assert profile.table_pages["lineitem"] == 13
        # orders: ceil(200 * 50 / 8192) = ceil(1.22) = 2
        assert profile.table_pages["orders"] == 2

    def test_from_explain_with_analyze(self):
        """Plan tree with block statistics (EXPLAIN ANALYZE)."""
        plan = {
            "Plan": {
                "Node Type": "Seq Scan",
                "Relation Name": "customer",
                "Plan Rows": 100,
                "Plan Width": 50,
                "Shared Hit Blocks": 30,
                "Shared Read Blocks": 10,
            }
        }
        profile = build_access_profile("q2", plan)
        assert profile.table_pages["customer"] == 40  # 30 + 10

    def test_duplicate_table_takes_max(self):
        """Self-join: same table appears twice, keep max pages."""
        plan = {
            "Plan": {
                "Node Type": "Merge Join",
                "Plans": [
                    {
                        "Node Type": "Seq Scan",
                        "Relation Name": "orders",
                        "Plan Rows": 1000,
                        "Plan Width": 80,
                    },
                    {
                        "Node Type": "Index Scan",
                        "Relation Name": "orders",
                        "Plan Rows": 100,
                        "Plan Width": 80,
                    },
                ],
            }
        }
        profile = build_access_profile("q_self_join", plan)
        # ceil(1000*80/8192) = 10, ceil(100*80/8192) = 1 → max = 10
        assert profile.table_pages["orders"] == 10

    def test_no_relations(self):
        """Plan with no table scans (e.g. SELECT 1)."""
        plan = {"Plan": {"Node Type": "Result"}}
        profile = build_access_profile("q_const", plan)
        assert profile.table_pages == {}
        assert profile.total_pages == 0
