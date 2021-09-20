# How to 1-Hot Encode
categorical_variables = [
  'make',
  'engine_fuel_type',
  'transmission_type',
  'driven_wheels',
  'market_category',
  'vehicle_size',
  'vehicle_style',
]
categories = {
 category: df[category].values_counts().head().index
   for category in categorical_variables
 }

 for category, values in categories.items():
   for value in values:
     feature = f"{category}_{value}"
     df[feature] = (df[category] == value).astype(int)
     features.append(feature)


