// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#pragma once

#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <thread>
#include <functional>

namespace omni::bridge {

class UnixDomainSocketIPC {
private:
    std::string socket_path_;
    int server_fd_;
    bool running_;
    std::thread listener_thread_;
    std::function<std::string(const std::string&)> message_handler_;

    void listen_loop() {
        while (running_) {
            struct sockaddr_un client_addr;
            socklen_t client_len = sizeof(client_addr);
            int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);
            
            if (client_fd < 0) {
                if (running_) {
                    std::cerr << "[IPC] Accept error: " << strerror(errno) << std::endl;
                }
                continue;
            }

            char buffer[4096];
            ssize_t bytes_read = read(client_fd, buffer, sizeof(buffer) - 1);
            if (bytes_read > 0) {
                buffer[bytes_read] = '\0';
                std::string request(buffer);
                
                // A handler exception must never kill the daemon (fail closed, log, continue)
                std::string response;
                try {
                    response = message_handler_ ? message_handler_(request) : "ACK";
                } catch (const std::exception& e) {
                    std::cerr << "[IPC] Handler exception (request dropped, daemon alive): "
                              << e.what() << std::endl;
                    response = std::string("{\"status\":\"error\",\"error\":\"handler_exception\"}");
                } catch (...) {
                    std::cerr << "[IPC] Unknown handler exception (request dropped, daemon alive)" << std::endl;
                    response = std::string("{\"status\":\"error\",\"error\":\"handler_exception\"}");
                }
                ssize_t ignored = write(client_fd, response.c_str(), response.size());
                (void)ignored;
            }
            close(client_fd);
        }
    }

public:
    UnixDomainSocketIPC(const std::string& socket_path) 
        : socket_path_(socket_path), server_fd_(-1), running_(false) {}

    ~UnixDomainSocketIPC() {
        stop();
    }

    void start(std::function<std::string(const std::string&)> handler) {
        message_handler_ = handler;
        
        server_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
        if (server_fd_ < 0) {
            throw std::runtime_error("Failed to create Unix Domain Socket");
        }

        // Unlink previous socket if it exists
        unlink(socket_path_.c_str());

        struct sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

        if (bind(server_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            close(server_fd_);
            throw std::runtime_error("Failed to bind Unix Domain Socket: " + std::string(strerror(errno)));
        }

        if (listen(server_fd_, 5) < 0) {
            close(server_fd_);
            throw std::runtime_error("Failed to listen on Unix Domain Socket");
        }

        running_ = true;
        listener_thread_ = std::thread(&UnixDomainSocketIPC::listen_loop, this);
        std::cout << "[IPC] Listening on " << socket_path_ << std::endl;
    }

    void stop() {
        if (running_) {
            running_ = false;
            shutdown(server_fd_, SHUT_RDWR);
            close(server_fd_);
            if (listener_thread_.joinable()) {
                listener_thread_.join();
            }
            unlink(socket_path_.c_str());
        }
    }

    // Client mode utility for testing
    static std::string send_message(const std::string& socket_path, const std::string& message, int timeout_sec = 5) {
        int sock = socket(AF_UNIX, SOCK_STREAM, 0);
        if (sock < 0) return "ERROR: Socket creation failed";

        // §57: never hang the bridge on an unresponsive agent — enforce SO_SNDTIMEO/SO_RCVTIMEO
        struct timeval tv{};
        tv.tv_sec = timeout_sec;
        setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        struct sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

        if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            close(sock);
            return "ERROR: Connection failed";
        }

        if (write(sock, message.c_str(), message.size()) < 0) {
            close(sock);
            return "ERROR: Write failed/timed out";
        }

        char buffer[4096];
        ssize_t bytes_read = read(sock, buffer, sizeof(buffer) - 1);
        close(sock);

        if (bytes_read >= 0) {
            buffer[bytes_read] = '\0';
            return std::string(buffer);
        }
        return "ERROR: Read failed/timed out";
    }
};

} // namespace omni::bridge
