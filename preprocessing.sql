-- TODO buffer

-- Create points table
CREATE TABLE IF NOT EXISTS points
(
    id numeric,
    x numeric, 
    y numeric,
    z numeric,
    geometry geometry(Point, 4326),
    final_geometry geometry(PointZ, 4326)
);

-- Create footprints table
CREATE TABLE IF NOT EXISTS footprints
(
    id numeric,
    geometry_text text,
    geometry geometry(MultiPolygon,4326),
    buffered_geometry geometry(Polygon,4326)
);

-- Copy points data from source path
COPY points (id, x, y, z)
FROM :points_source_path
WITH (FORMAT CSV, HEADER true);

update points set geometry = ST_MakePoint(x, y), final_geometry = ST_MakePoint(x, y, z);

-- Copy footprints data from source path
COPY footprints (id, geometry_text)
FROM :footprints_source_path
WITH (FORMAT CSV, HEADER true);

update footprints set geometry = ST_Multi(ST_GeomFromText(geometry_text));
update footprints set buffered_geometry = ST_Buffer(geometry, 0.00005);

-- Generate and export output to a csv file
COPY (
    select footprint_id, 
            footprint_geometry, 
            array_agg(point_id) AS points_id,
            ST_AsText(ST_Collect(point_geometry)) AS points_geometry
    from (
        SELECT f.id as footprint_id,
                ST_AsText(f.geometry) as footprint_geometry, 
                p.id as point_id,
                p.final_geometry as point_geometry
        FROM footprints f
        INNER JOIN points p on ST_Intersects(f.buffered_geometry, p.geometry)
        ) joined_data 
    group by footprint_id, footprint_geometry)
TO :output_path
WITH (FORMAT CSV, HEADER true);
