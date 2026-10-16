// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include <string>
#include <filesystem>
#include <fstream>
#include <system_error>

namespace omni::bridge {

class CgroupV2Manager {
private:
    std::filesystem::path base_path_;

public:
    explicit CgroupV2Manager(std::string base_path = "/sys/fs/cgroup/omni-bridge") 
        : base_path_(std::move(base_path)) {
        std::error_code ec;
        std::filesystem::create_directories(base_path_, ec);
    }

    std::string create_sandbox(const std::string& sandbox_id) const {
        auto path = base_path_ / sandbox_id;
        std::error_code ec;
        std::filesystem::create_directories(path, ec);
        return path.string();
    }

    bool attach_process(const std::string& sandbox_id, pid_t pid) const noexcept {
        auto path = base_path_ / sandbox_id / "cgroup.procs";
        std::ofstream ofs(path);
        if (!ofs) return false;
        ofs << pid;
        return true;
    }

    bool freeze(const std::string& sandbox_id) const noexcept {
        auto path = base_path_ / sandbox_id / "cgroup.freeze";
        std::ofstream ofs(path);
        if (!ofs) return false;
        ofs << "1";
        return true;
    }

    bool kill_all(const std::string& sandbox_id) const noexcept {
        auto path = base_path_ / sandbox_id / "cgroup.kill";
        std::ofstream ofs(path);
        if (!ofs) return false;
        ofs << "1";
        return true;
    }
};

} // namespace omni::bridge
