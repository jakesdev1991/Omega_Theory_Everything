// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include <sqlite3.h>
#include <string>
#include <vector>
#include <stdexcept>

namespace omni::bridge {

struct MemoryRecord {
    int id;
    std::string session_id;
    std::string agent_role;
    std::string content;
    std::string metadata;
};

class SQLiteMemoryEngine {
private:
    sqlite3* db_{nullptr};

    void execute_query(const std::string& query) {
        char* err_msg = nullptr;
        if (sqlite3_exec(db_, query.c_str(), nullptr, nullptr, &err_msg) != SQLITE_OK) {
            std::string err = err_msg;
            sqlite3_free(err_msg);
            throw std::runtime_error("SQLite error: " + err);
        }
    }

public:
    explicit SQLiteMemoryEngine(const std::string& db_path = ":memory:") {
        if (sqlite3_open(db_path.c_str(), &db_) != SQLITE_OK) {
            throw std::runtime_error("Failed to open SQLite DB: " + std::string(sqlite3_errmsg(db_)));
        }

        std::string init_query = 
            "CREATE TABLE IF NOT EXISTS history ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "session_id TEXT, "
            "agent_role TEXT, "
            "content TEXT, "
            "metadata TEXT, "
            "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)";
        
        execute_query(init_query);
    }

    ~SQLiteMemoryEngine() {
        if (db_) {
            sqlite3_close(db_);
        }
    }

    void save(const std::string& session_id, const std::string& role, const std::string& content, const std::string& meta = "{}") {
        sqlite3_stmt* stmt;
        std::string query = "INSERT INTO history (session_id, agent_role, content, metadata) VALUES (?, ?, ?, ?)";
        
        if (sqlite3_prepare_v2(db_, query.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
            throw std::runtime_error("Failed to prepare insert statement");
        }

        sqlite3_bind_text(stmt, 1, session_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, role.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3, content.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 4, meta.c_str(), -1, SQLITE_TRANSIENT);

        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }
};

} // namespace omni::bridge
