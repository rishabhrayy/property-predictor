// Builds the model's feature vector in the browser. Mirrors src/features/build.py exactly:
// same order, same log transforms, same fill values; tests/test_pipeline.py checks this.
export const MONTHS_SINCE_2016 = 26; // March 2018, the end of the data

export function buildFeatures(meta, input, months = MONTHS_SINCE_2016) {
  const { spec } = meta;
  const f = spec.fill;
  const geo = spec.suburb_geo[input.suburb];
  const subMed = spec.suburb_median_log[input.suburb] ?? spec.global_median_log;
  const subTypeMed =
    spec.suburb_type_median_log[`${input.suburb}|${input.type}`] ?? subMed ?? spec.type_median_log[input.type];
  const x = {
    rooms: input.rooms,
    bathrooms: input.bath ?? f.bathrooms,
    car: input.car ?? f.car,
    landsize_log: input.land != null ? Math.log1p(input.land) : f.landsize_log,
    building_area_log: input.building != null ? Math.log1p(input.building) : f.building_area_log,
    year_built: input.year ?? f.year_built,
    type_h: input.type === 'h' ? 1 : 0,
    type_t: input.type === 't' ? 1 : 0,
    type_u: input.type === 'u' ? 1 : 0,
    distance_km: geo?.dist ?? f.distance_km,
    latitude: geo?.lat ?? f.latitude,
    longitude: geo?.lon ?? f.longitude,
    suburb_median_log: subMed,
    suburb_type_median_log: subTypeMed,
    months_since_2016: months,
  };
  return { vector: meta.features.map((name) => x[name]), baseline: Math.exp(subTypeMed) };
}
