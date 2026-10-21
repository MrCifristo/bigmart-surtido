import pandas as pd
import pytest


@pytest.fixture
def raw_bigmart() -> pd.DataFrame:
    """Muestra pequeña con los problemas de calidad del dataset real."""
    return pd.DataFrame(
        {
            "Item_Identifier": ["FDA15", "FDA15", "DRC01", "NCD19", "NCD19", "FDX07"],
            "Item_Weight": [9.3, None, 5.92, 8.93, 8.93, None],
            "Item_Fat_Content": [
                "Low Fat",
                "LF",
                "reg",
                "Low Fat",
                "low fat",
                "Regular",
            ],
            "Item_Visibility": [0.016, 0.0, 0.019, 0.0, 0.05, 0.3],
            "Item_Type": [
                "Dairy",
                "Dairy",
                "Soft Drinks",
                "Household",
                "Household",
                "Fruits and Vegetables",
            ],
            "Item_MRP": [249.8, 250.1, 48.3, 53.9, 54.1, 182.1],
            "Outlet_Identifier": [
                "OUT049",
                "OUT027",
                "OUT018",
                "OUT013",
                "OUT017",
                "OUT010",
            ],
            "Outlet_Establishment_Year": [1999, 1985, 2009, 1987, 2007, 1998],
            "Outlet_Size": ["Medium", "Medium", "Medium", "High", None, None],
            "Outlet_Location_Type": [
                "Tier 1",
                "Tier 3",
                "Tier 3",
                "Tier 3",
                "Tier 2",
                "Tier 3",
            ],
            "Outlet_Type": [
                "Supermarket Type1",
                "Supermarket Type3",
                "Supermarket Type2",
                "Supermarket Type1",
                "Supermarket Type1",
                "Grocery Store",
            ],
            "Item_Outlet_Sales": [3735.1, 4022.8, 443.4, 994.7, 1100.0, 732.4],
        }
    )
