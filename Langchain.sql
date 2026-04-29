--
-- PostgreSQL database dump
--

\restrict pAigZvDXVneOdDmNlATF2lVbynDu0lI1IaWOUaSRSg1vb120Hv7JXQT7THrj7LD

-- Dumped from database version 18.0 (Debian 18.0-1.pgdg13+3)
-- Dumped by pg_dump version 18.0 (Debian 18.0-1.pgdg13+3)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: vector; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;


--
-- Name: EXTENSION vector; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION vector IS 'vector data type and ivfflat and hnsw access methods';


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: alembic_version; Type: TABLE; Schema: public; Owner: guest
--

CREATE TABLE public.alembic_version (
    version_num character varying(32) NOT NULL
);


ALTER TABLE public.alembic_version OWNER TO guest;

--
-- Name: checkpoint_blobs; Type: TABLE; Schema: public; Owner: guest
--

CREATE TABLE public.checkpoint_blobs (
    thread_id text NOT NULL,
    checkpoint_ns text DEFAULT ''::text NOT NULL,
    channel text NOT NULL,
    version text NOT NULL,
    type text NOT NULL,
    blob bytea
);


ALTER TABLE public.checkpoint_blobs OWNER TO guest;

--
-- Name: checkpoint_migrations; Type: TABLE; Schema: public; Owner: guest
--

CREATE TABLE public.checkpoint_migrations (
    v integer NOT NULL
);


ALTER TABLE public.checkpoint_migrations OWNER TO guest;

--
-- Name: checkpoint_writes; Type: TABLE; Schema: public; Owner: guest
--

CREATE TABLE public.checkpoint_writes (
    thread_id text NOT NULL,
    checkpoint_ns text DEFAULT ''::text NOT NULL,
    checkpoint_id text NOT NULL,
    task_id text NOT NULL,
    idx integer NOT NULL,
    channel text NOT NULL,
    type text,
    blob bytea NOT NULL,
    task_path text DEFAULT ''::text NOT NULL
);


ALTER TABLE public.checkpoint_writes OWNER TO guest;

--
-- Name: checkpoints; Type: TABLE; Schema: public; Owner: guest
--

CREATE TABLE public.checkpoints (
    thread_id text NOT NULL,
    checkpoint_ns text DEFAULT ''::text NOT NULL,
    checkpoint_id text NOT NULL,
    parent_checkpoint_id text,
    type text,
    checkpoint jsonb NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb NOT NULL
);


ALTER TABLE public.checkpoints OWNER TO guest;

--
-- Name: ragagent_vectorstore; Type: TABLE; Schema: public; Owner: guest
--

CREATE TABLE public.ragagent_vectorstore (
    langchain_id uuid NOT NULL,
    content text NOT NULL,
    embedding public.vector(768) NOT NULL,
    langchain_metadata json
);


ALTER TABLE public.ragagent_vectorstore OWNER TO guest;

--
-- Name: store; Type: TABLE; Schema: public; Owner: guest
--

CREATE TABLE public.store (
    prefix text NOT NULL,
    key text NOT NULL,
    value jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    expires_at timestamp with time zone,
    ttl_minutes integer
);


ALTER TABLE public.store OWNER TO guest;

--
-- Name: store_migrations; Type: TABLE; Schema: public; Owner: guest
--

CREATE TABLE public.store_migrations (
    v integer NOT NULL
);


ALTER TABLE public.store_migrations OWNER TO guest;

--
-- Name: store_vectors; Type: TABLE; Schema: public; Owner: guest
--

CREATE TABLE public.store_vectors (
    prefix text NOT NULL,
    key text NOT NULL,
    field_name text NOT NULL,
    embedding public.vector(768),
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.store_vectors OWNER TO guest;

--
-- Name: users; Type: TABLE; Schema: public; Owner: guest
--

CREATE TABLE public.users (
    id integer NOT NULL,
    firstname character varying(128) NOT NULL,
    lastname character varying(128) NOT NULL,
    email character varying(255) NOT NULL,
    phone character varying(15),
    password character varying(128),
    lastlogin timestamp with time zone,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    modified_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.users OWNER TO guest;

--
-- Name: users_id_seq; Type: SEQUENCE; Schema: public; Owner: guest
--

CREATE SEQUENCE public.users_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.users_id_seq OWNER TO guest;

--
-- Name: users_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: guest
--

ALTER SEQUENCE public.users_id_seq OWNED BY public.users.id;


--
-- Name: vector_migrations; Type: TABLE; Schema: public; Owner: guest
--

CREATE TABLE public.vector_migrations (
    v integer NOT NULL
);


ALTER TABLE public.vector_migrations OWNER TO guest;

--
-- Name: users id; Type: DEFAULT; Schema: public; Owner: guest
--

ALTER TABLE ONLY public.users ALTER COLUMN id SET DEFAULT nextval('public.users_id_seq'::regclass);


--
-- Data for Name: alembic_version; Type: TABLE DATA; Schema: public; Owner: guest
--

COPY public.alembic_version (version_num) FROM stdin;
12c3cd5b0d47
\.


--
-- Data for Name: checkpoint_blobs; Type: TABLE DATA; Schema: public; Owner: guest
--

COPY public.checkpoint_blobs (thread_id, checkpoint_ns, channel, version, type, blob) FROM stdin;
\.


--
-- Data for Name: checkpoint_migrations; Type: TABLE DATA; Schema: public; Owner: guest
--

COPY public.checkpoint_migrations (v) FROM stdin;
0
1
2
3
4
5
6
7
8
9
\.


--
-- Data for Name: checkpoint_writes; Type: TABLE DATA; Schema: public; Owner: guest
--

COPY public.checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob, task_path) FROM stdin;
\.


--
-- Data for Name: checkpoints; Type: TABLE DATA; Schema: public; Owner: guest
--

COPY public.checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) FROM stdin;
\.


--
-- Data for Name: ragagent_vectorstore; Type: TABLE DATA; Schema: public; Owner: guest
--

COPY public.ragagent_vectorstore (langchain_id, content, embedding, langchain_metadata) FROM stdin;
\.


--
-- Data for Name: store; Type: TABLE DATA; Schema: public; Owner: guest
--

COPY public.store (prefix, key, value, created_at, updated_at, expires_at, ttl_minutes) FROM stdin;
\.


--
-- Data for Name: store_migrations; Type: TABLE DATA; Schema: public; Owner: guest
--

COPY public.store_migrations (v) FROM stdin;
0
1
2
3
\.


--
-- Data for Name: store_vectors; Type: TABLE DATA; Schema: public; Owner: guest
--

COPY public.store_vectors (prefix, key, field_name, embedding, created_at, updated_at) FROM stdin;
\.


--
-- Data for Name: users; Type: TABLE DATA; Schema: public; Owner: guest
--

COPY public.users (id, firstname, lastname, email, phone, password, lastlogin, created_at, modified_at) FROM stdin;
1	Kok How	Teh	khteh@email.com	\N	$2b$10$fBHPNGkND06ksrmTULGAoewTRPuezEuJwrV9JFLQvDMGa.BC0fgQi	\N	2026-04-29 11:24:30.320056+00	2026-04-29 11:24:30.320056+00
2	Mickey	Mouse	mickey@email.com	\N	$2b$10$zZ/95mFOS7.wMBfxPsbCYu3JQ3Y7X62thZToU0XX73IeOk0XXej9y	\N	2026-04-29 11:24:50.272816+00	2026-04-29 11:24:50.272816+00
\.


--
-- Data for Name: vector_migrations; Type: TABLE DATA; Schema: public; Owner: guest
--

COPY public.vector_migrations (v) FROM stdin;
0
1
2
\.


--
-- Name: users_id_seq; Type: SEQUENCE SET; Schema: public; Owner: guest
--

SELECT pg_catalog.setval('public.users_id_seq', 2, true);


--
-- Name: alembic_version alembic_version_pkc; Type: CONSTRAINT; Schema: public; Owner: guest
--

ALTER TABLE ONLY public.alembic_version
    ADD CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num);


--
-- Name: checkpoint_blobs checkpoint_blobs_pkey; Type: CONSTRAINT; Schema: public; Owner: guest
--

ALTER TABLE ONLY public.checkpoint_blobs
    ADD CONSTRAINT checkpoint_blobs_pkey PRIMARY KEY (thread_id, checkpoint_ns, channel, version);


--
-- Name: checkpoint_migrations checkpoint_migrations_pkey; Type: CONSTRAINT; Schema: public; Owner: guest
--

ALTER TABLE ONLY public.checkpoint_migrations
    ADD CONSTRAINT checkpoint_migrations_pkey PRIMARY KEY (v);


--
-- Name: checkpoint_writes checkpoint_writes_pkey; Type: CONSTRAINT; Schema: public; Owner: guest
--

ALTER TABLE ONLY public.checkpoint_writes
    ADD CONSTRAINT checkpoint_writes_pkey PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx);


--
-- Name: checkpoints checkpoints_pkey; Type: CONSTRAINT; Schema: public; Owner: guest
--

ALTER TABLE ONLY public.checkpoints
    ADD CONSTRAINT checkpoints_pkey PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id);


--
-- Name: ragagent_vectorstore ragagent_vectorstore_pkey; Type: CONSTRAINT; Schema: public; Owner: guest
--

ALTER TABLE ONLY public.ragagent_vectorstore
    ADD CONSTRAINT ragagent_vectorstore_pkey PRIMARY KEY (langchain_id);


--
-- Name: store_migrations store_migrations_pkey; Type: CONSTRAINT; Schema: public; Owner: guest
--

ALTER TABLE ONLY public.store_migrations
    ADD CONSTRAINT store_migrations_pkey PRIMARY KEY (v);


--
-- Name: store store_pkey; Type: CONSTRAINT; Schema: public; Owner: guest
--

ALTER TABLE ONLY public.store
    ADD CONSTRAINT store_pkey PRIMARY KEY (prefix, key);


--
-- Name: store_vectors store_vectors_pkey; Type: CONSTRAINT; Schema: public; Owner: guest
--

ALTER TABLE ONLY public.store_vectors
    ADD CONSTRAINT store_vectors_pkey PRIMARY KEY (prefix, key, field_name);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: guest
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- Name: vector_migrations vector_migrations_pkey; Type: CONSTRAINT; Schema: public; Owner: guest
--

ALTER TABLE ONLY public.vector_migrations
    ADD CONSTRAINT vector_migrations_pkey PRIMARY KEY (v);


--
-- Name: checkpoint_blobs_thread_id_idx; Type: INDEX; Schema: public; Owner: guest
--

CREATE INDEX checkpoint_blobs_thread_id_idx ON public.checkpoint_blobs USING btree (thread_id);


--
-- Name: checkpoint_writes_thread_id_idx; Type: INDEX; Schema: public; Owner: guest
--

CREATE INDEX checkpoint_writes_thread_id_idx ON public.checkpoint_writes USING btree (thread_id);


--
-- Name: checkpoints_thread_id_idx; Type: INDEX; Schema: public; Owner: guest
--

CREATE INDEX checkpoints_thread_id_idx ON public.checkpoints USING btree (thread_id);


--
-- Name: idx_store_expires_at; Type: INDEX; Schema: public; Owner: guest
--

CREATE INDEX idx_store_expires_at ON public.store USING btree (expires_at) WHERE (expires_at IS NOT NULL);


--
-- Name: ix_users_email; Type: INDEX; Schema: public; Owner: guest
--

CREATE UNIQUE INDEX ix_users_email ON public.users USING btree (email);


--
-- Name: ix_users_phone; Type: INDEX; Schema: public; Owner: guest
--

CREATE UNIQUE INDEX ix_users_phone ON public.users USING btree (phone);


--
-- Name: store_prefix_idx; Type: INDEX; Schema: public; Owner: guest
--

CREATE INDEX store_prefix_idx ON public.store USING btree (prefix text_pattern_ops);


--
-- Name: store_vectors_embedding_idx; Type: INDEX; Schema: public; Owner: guest
--

CREATE INDEX store_vectors_embedding_idx ON public.store_vectors USING hnsw (embedding public.vector_cosine_ops);


--
-- Name: store_vectors store_vectors_prefix_key_fkey; Type: FK CONSTRAINT; Schema: public; Owner: guest
--

ALTER TABLE ONLY public.store_vectors
    ADD CONSTRAINT store_vectors_prefix_key_fkey FOREIGN KEY (prefix, key) REFERENCES public.store(prefix, key) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

\unrestrict pAigZvDXVneOdDmNlATF2lVbynDu0lI1IaWOUaSRSg1vb120Hv7JXQT7THrj7LD

