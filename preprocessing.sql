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
    edifc_stat text, 
    edifc_ty text, 
    edifc_uso text,
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
COPY footprints (id, edifc_stat, edifc_ty, edifc_uso, geometry_text)
FROM :footprints_source_path
WITH (FORMAT CSV, HEADER true);

update footprints set geometry = ST_Multi(ST_GeomFromText(geometry_text));
update footprints set buffered_geometry = ST_Buffer(geometry, 0.00008);

-- Generate and export output to a csv file
COPY (
    select footprint_id, 
		edifc_stat,
		edifc_ty,
		edifc_uso,
		max_in_footprint,
		percentile_20_in_footprint,
		percentile_40_in_footprint,
		percentile_60_in_footprint,
		percentile_80_in_footprint,
		min_overrall,
		-- Height of building (Max above - Min overrall)
		max_in_footprint - min_overrall as building_height,
		-- Height of roof (Max above - Min above)
		max_in_footprint - percentile_40_in_footprint as roof_height,
		footprint_area,
		footprint_geometry,
		points_geometry
	from  ( 
		select footprint_id, 
				edifc_stat,
				edifc_ty,
				edifc_uso,
				-- Max within the footprint
				MAX( CASE WHEN ST_Intersects(footprint_geometry, point_geometry_2d) THEN point_height END) as max_in_footprint,
				-- Percentiles within the footprint
				percentile_cont(.2) within group (order by point_height asc) 
					filter (where ST_Intersects(footprint_geometry, point_geometry_2d)) as percentile_20_in_footprint,
				percentile_cont(.4) within group (order by point_height asc) 
					filter (where ST_Intersects(footprint_geometry, point_geometry_2d)) as percentile_40_in_footprint,
				percentile_cont(.6) within group (order by point_height asc) 
					filter (where ST_Intersects(footprint_geometry, point_geometry_2d)) as percentile_60_in_footprint,
				percentile_cont(.8) within group (order by point_height asc)
					filter (where ST_Intersects(footprint_geometry, point_geometry_2d)) as percentile_80_in_footprint,
				-- Min overrall
				MIN(point_height) as min_overrall,
				-- Area of footprint
				ST_Area(ST_Transform(footprint_geometry, utmzone(ST_Centroid(footprint_geometry)))) as footprint_area,
				-- TODO Slope on a grid over the footprint
				-- TODO Numer of relative Max/Min in footprint
				-- TODO Proximity of other footprints?
				ST_AsText(footprint_geometry) as footprint_geometry, 
				ST_AsText(ST_Collect(point_geometry)) AS points_geometry
		from (
			SELECT f.id as footprint_id,
					edifc_stat,
					edifc_ty,
					edifc_uso,
					ST_Z(p.final_geometry) as point_height, 
					f.geometry as footprint_geometry, 
					p.geometry as point_geometry_2d, 
					p.final_geometry as point_geometry
			FROM footprints f
			INNER JOIN points p on ST_Intersects(f.buffered_geometry, p.geometry)
        ) intersection_ 
        group by footprint_id, edifc_stat, edifc_ty, edifc_uso, footprint_geometry
    ) joined_data)
TO :output_path
WITH (FORMAT CSV, HEADER true);
