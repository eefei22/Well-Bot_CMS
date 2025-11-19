-- Comprehensive fix: Drop ALL variations of match_embeddings and create ONE definitive version
-- This resolves function overloading ambiguity issues

-- Step 1: Drop ALL possible variations of match_embeddings function
-- We need to drop by signature (parameter types), not by name

-- Drop version without match_limit/index_limit
DROP FUNCTION IF EXISTS match_embeddings(vector(768), uuid, text, text, double precision);

-- Drop version with match_limit/index_limit (order 1: match_limit before match_threshold)
DROP FUNCTION IF EXISTS match_embeddings(vector(768), uuid, text, text, integer, double precision, integer);

-- Drop version with match_limit/index_limit (order 2: match_threshold before match_limit)  
DROP FUNCTION IF EXISTS match_embeddings(vector(768), uuid, text, text, double precision, integer, integer);

-- Drop any other variations that might exist
-- Using CASCADE to drop dependent objects if any
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN 
        SELECT oid, proname, pg_get_function_identity_arguments(oid) as args
        FROM pg_proc
        WHERE proname = 'match_embeddings'
        AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
    LOOP
        EXECUTE 'DROP FUNCTION IF EXISTS ' || quote_ident(r.proname) || '(' || r.args || ') CASCADE';
        RAISE NOTICE 'Dropped function: %(%)', r.proname, r.args;
    END LOOP;
END $$;

-- Step 2: Create ONE definitive version with parameters in the exact order we call from Python
-- Parameter order: query_vector, match_user_id, match_model_tag, match_kind, match_threshold, match_limit, index_limit
CREATE OR REPLACE FUNCTION match_embeddings(
  query_vector vector(768),
  match_user_id uuid,
  match_model_tag text,
  match_kind text DEFAULT 'message',
  match_threshold double precision DEFAULT 0.7,
  match_limit integer DEFAULT NULL,
  index_limit integer DEFAULT 100
)
RETURNS TABLE (
  ref_id uuid,
  similarity double precision,
  kind text,
  created_at timestamptz
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT 
    emb.ref_id,
    GREATEST(0, LEAST(1, 1 - (emb.vector <=> query_vector)))::double precision as similarity,
    emb.kind,
    emb.created_at
  FROM (
    SELECT 
      wb_embeddings.ref_id,
      wb_embeddings.vector,
      wb_embeddings.kind,
      wb_embeddings.created_at
    FROM wb_embeddings
    WHERE wb_embeddings.user_id = match_user_id
      AND wb_embeddings.model_tag = match_model_tag
      AND wb_embeddings.kind = match_kind
    ORDER BY wb_embeddings.vector <=> query_vector
    LIMIT index_limit
  ) emb
  WHERE 1 - (emb.vector <=> query_vector) >= match_threshold
  ORDER BY emb.vector <=> query_vector
  LIMIT COALESCE(match_limit, 2147483647);  -- No limit if NULL
END;
$$;

-- Verify the function was created
DO $$
DECLARE
    func_count integer;
BEGIN
    SELECT COUNT(*) INTO func_count
    FROM pg_proc
    WHERE proname = 'match_embeddings'
    AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public');
    
    IF func_count = 1 THEN
        RAISE NOTICE 'SUCCESS: Exactly one match_embeddings function exists';
    ELSE
        RAISE WARNING 'WARNING: Found % match_embeddings functions (expected 1)', func_count;
    END IF;
END $$;

