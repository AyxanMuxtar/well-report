-- =============================================================================
-- Drilling daily report schema (DuckDB)
-- =============================================================================
-- Tables:
--   reports          : one row per PDF, header + summary fields
--   operations       : one row per operation entry within a report
--   drilling_fluid   : drilling fluid samples
--   pore_pressure    : pore pressure readings
--   survey_station   : directional survey stations
--   lithology        : lithology intervals
--   gas_reading      : gas reading classes/ranges
--   parse_errors     : log of PDFs that failed to parse (for QA)
-- =============================================================================

DROP TABLE IF EXISTS gas_reading;
DROP TABLE IF EXISTS lithology;
DROP TABLE IF EXISTS survey_station;
DROP TABLE IF EXISTS pore_pressure;
DROP TABLE IF EXISTS drilling_fluid;
DROP TABLE IF EXISTS operations;
DROP TABLE IF EXISTS reports;
DROP TABLE IF EXISTS parse_errors;

-- =============================================================================
-- reports : one row per PDF
-- =============================================================================
CREATE TABLE reports (
    pdf_path                          VARCHAR PRIMARY KEY,        -- absolute or repo-relative path
    pdf_filename                      VARCHAR NOT NULL,
    well_prefix                       VARCHAR,                    -- e.g. "15_9_F_15_A"
    well_family                       VARCHAR,                    -- pooled, e.g. "15_9_F_15"
    wellbore_id                       VARCHAR,                    -- e.g. "15/9-F-15 A"  (from PDF body)
    report_number                     INTEGER,
    report_date                       DATE,                       -- from filename
    period_start                      TIMESTAMP,
    period_end                        TIMESTAMP,
    status                            VARCHAR,
    report_creation_time              TIMESTAMP,
    days_ahead_behind                 DOUBLE,

    operator                          VARCHAR,
    rig_name                          VARCHAR,
    drilling_contractor               VARCHAR,
    spud_date                         TIMESTAMP,
    wellbore_type                     VARCHAR,
    date_well_complete                DATE,

    elevation_rkb_msl_m               DOUBLE,
    water_depth_msl_m                 DOUBLE,
    tight_well                        BOOLEAN,
    hpht                              BOOLEAN,
    temperature                       DOUBLE,
    pressure                          DOUBLE,

    dist_drilled_m                    DOUBLE,
    penetration_rate_mph              DOUBLE,
    hole_dia_in                       DOUBLE,
    pressure_test_type                VARCHAR,
    formation_strength_gcm3           DOUBLE,
    dia_last_casing                   DOUBLE,

    depth_kickoff_md                  DOUBLE,
    depth_kickoff_tvd                 DOUBLE,
    depth_md                          DOUBLE,
    depth_tvd                         DOUBLE,
    plug_back_depth_md                DOUBLE,
    depth_formation_strength_md       DOUBLE,
    depth_formation_strength_tvd      DOUBLE,
    depth_last_casing_md              DOUBLE,
    depth_last_casing_tvd             DOUBLE,

    summary_24h                       TEXT,
    planned_24h                       TEXT,

    parse_quality                     VARCHAR,                    -- 'full' | 'partial' | 'header_only'
    ingested_at                       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- operations : one row per operation entry
-- =============================================================================
CREATE TABLE operations (
    op_id                             BIGINT PRIMARY KEY,
    pdf_path                          VARCHAR NOT NULL,           -- FK to reports
    well_family                       VARCHAR,                    -- denormalized for fast filtering
    well_prefix                       VARCHAR,
    report_date                       DATE,
    op_index                          INTEGER,                    -- order within the report
    start_time                        VARCHAR,                    -- "HH:MM" as text (some are missing)
    end_time                          VARCHAR,
    end_depth_md                      DOUBLE,
    main_activity                     VARCHAR,
    sub_activity                      VARCHAR,
    state                             VARCHAR,                    -- 'ok' | 'fail' | etc
    remark                            TEXT,
    op_text                           TEXT                        -- precomputed: main + sub + state + remark (for matching)
);

CREATE INDEX idx_operations_well_family ON operations(well_family);
CREATE INDEX idx_operations_pdf_path    ON operations(pdf_path);

-- =============================================================================
-- drilling_fluid : drilling fluid samples
-- =============================================================================
CREATE TABLE drilling_fluid (
    fluid_id                          BIGINT PRIMARY KEY,
    pdf_path                          VARCHAR NOT NULL,
    well_family                       VARCHAR,
    sample_index                      INTEGER,
    sample_time                       VARCHAR,
    sample_point                      VARCHAR,
    sample_depth_md                   DOUBLE,
    fluid_type                        VARCHAR,
    fluid_density_gcm3                DOUBLE,
    funnel_visc_s                     DOUBLE,
    plastic_visc_mpas                 DOUBLE,
    yield_point_pa                    DOUBLE,
    test_temp_hpht_degc               DOUBLE
);

-- =============================================================================
-- pore_pressure
-- =============================================================================
CREATE TABLE pore_pressure (
    pp_id                             BIGINT PRIMARY KEY,
    pdf_path                          VARCHAR NOT NULL,
    well_family                       VARCHAR,
    reading_index                     INTEGER,
    sample_time                       VARCHAR,
    depth_md                          DOUBLE,
    depth_tvd                         DOUBLE,
    equ_mud_weight_gcm3               DOUBLE,
    reading_type                      VARCHAR
);

-- =============================================================================
-- survey_station
-- =============================================================================
CREATE TABLE survey_station (
    survey_id                         BIGINT PRIMARY KEY,
    pdf_path                          VARCHAR NOT NULL,
    well_family                       VARCHAR,
    station_index                     INTEGER,
    depth_md                          DOUBLE,
    depth_tvd                         DOUBLE,
    inclination_deg                   DOUBLE,
    azimuth_deg                       DOUBLE,
    comment                           TEXT
);

-- =============================================================================
-- lithology
-- =============================================================================
CREATE TABLE lithology (
    litho_id                          BIGINT PRIMARY KEY,
    pdf_path                          VARCHAR NOT NULL,
    well_family                       VARCHAR,
    interval_index                    INTEGER,
    start_depth_md                    DOUBLE,
    end_depth_md                      DOUBLE,
    start_depth_tvd                   DOUBLE,
    end_depth_tvd                     DOUBLE,
    shows_description                 TEXT,
    lithology_description             TEXT
);

-- =============================================================================
-- gas_reading
-- =============================================================================
CREATE TABLE gas_reading (
    gas_id                            BIGINT PRIMARY KEY,
    pdf_path                          VARCHAR NOT NULL,
    well_family                       VARCHAR,
    reading_index                     INTEGER,
    sample_time                       VARCHAR,
    gas_class                         VARCHAR,
    depth_top_md                      DOUBLE,
    depth_bottom_md                   DOUBLE,
    depth_top_tvd                     DOUBLE,
    depth_bottom_tvd                  DOUBLE,
    c1_ppm                            DOUBLE,
    c2_ppm                            DOUBLE,
    c3_ppm                            DOUBLE,
    ic4_ppm                           DOUBLE,
    ic5_ppm                           DOUBLE,
    highest_gas_pct                   DOUBLE,
    lowest_gas_pct                    DOUBLE
);

-- =============================================================================
-- parse_errors : log of PDFs that failed (for diagnostics)
-- =============================================================================
CREATE TABLE parse_errors (
    pdf_path                          VARCHAR NOT NULL,
    stage                             VARCHAR,                    -- 'header' | 'operations' | 'fluid' | etc
    error_message                     TEXT,
    occurred_at                       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
