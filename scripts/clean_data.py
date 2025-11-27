
import pandas as pd
import numpy as np

def clean_immo_data(file_path):

    # 1️⃣ Veri oku
    df = pd.read_csv(file_path, encoding="utf-8")
    
    # 2️⃣ Duplicate kontrolü
    df = df.drop_duplicates()
    
    # 3️⃣ Whitespace check
    has_whitespace = df.applymap(lambda x: isinstance(x, str) and x != x.strip()).any().any()
    print("Contains whitespace?", has_whitespace)
    
    # 4️⃣ Province ve Region fonksiyonları
    provinces = {
        "brussels": list(range(1000, 1300)),
        "brabant_walloon": list(range(1300, 1500)),
        "brabant_flemish": list(range(1500, 2000)) + list(range(3000, 3500)),
        "antwerp": list(range(2000, 3000)),
        "limburg": list(range(3500, 4000)),
        "liege": list(range(4000, 4500)),
        "namur": list(range(4500, 5681)),
        "hainaut": list(range(5681, 6600)) + list(range(7000, 8000)),
        "luxembourg": list(range(6600, 7000)),
        "west_flanders": list(range(8000, 9000)),
        "east_flanders": list(range(9000, 10000))
    }
    
    def postcode_to_region(pc):
        if pd.isna(pc):
            return "Unknown"
        pc = int(pc)
        if 1000 <= pc <= 1299:
            return "Brussels"
        if 1300 <= pc <= 1499 or 4000 <= pc <= 7999:
            return "Wallonia"
        if 1500 <= pc <= 3999 or 8000 <= pc <= 9999:
            return "Flanders"
        return "Unknown"
    
    def postcode_to_province(postcode):
        for prov, codes in provinces.items():
            if postcode in codes:
                return prov.replace('_', ' ').title()
        return "Unknown"
    
    df["Region"] = df["postal_code"].apply(postcode_to_region)
    df["province"] = df["postal_code"].apply(postcode_to_province)
    
    # 5️⃣ Price per sqm hesaplamaları
    df["price_per_sqm"] = df["Price"] / df["Livable surface"]
    df['Price_per_sqm_land'] = df['Price'] / df['Total land surface']
    
    # 6️⃣ main_type normalize etme
    apartment_subtypes = ["apartment", "ground floor", "penthouse", "studio", "duplex", "loft", "triplex", "student flat"]
    house_subtypes = ["Residence", "Villa", "Mixed Building", "Master House", "Cottage", "Bungalow", "Chalet", "Mansion"]
    business_subtypes = ['commercial building', 'industrial building', 'office space', 'business surface', 'garage', 'parking']
    investment_subtypes = ['investment property']
    land_subtypes = ['land', 'development site', 'farming site', 'farmland', 'industrial ground', 'wood', 'green zone', 'recreational land', 'to parcel out site']
    
    def normalize_string(s):
        return str(s).strip().lower().replace('-', ' ')
    
    mapping = {normalize_string(x): "apartment" for x in apartment_subtypes}
    mapping.update({normalize_string(x): "house" for x in house_subtypes})
    mapping.update({normalize_string(x): "business" for x in business_subtypes})
    mapping.update({normalize_string(x): "investment property" for x in investment_subtypes})
    mapping.update({normalize_string(x): "land" for x in land_subtypes})
    
    df['main_type'] = df['type'].apply(lambda t: mapping.get(normalize_string(t), t))
    
    # Unmatched log
    unmapped = df[df['main_type'] == df['type']]['type'].unique()
    print("Not matched types:", unmapped)
    
    # 7️⃣ Dosyayı kaydet
    df.to_csv(file_path, index=False, encoding="utf-8")
    
    return df


if __name__ == "__main__":
    file_path = r"C:\Users\esram\OneDrive\Desktop\esrabecode\immo-eliza-ml\data\immovlan_cleaned_file_final.csv"
    df = clean_immo_data(file_path)
