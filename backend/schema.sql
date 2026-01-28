CREATE TABLE chat_messages (
    id BIGSERIAL PRIMARY KEY,

    session_id VARCHAR(128) NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),

    content TEXT NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
