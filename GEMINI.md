# AGENT.md

## Purpose

This document defines the engineering standards, architecture, and workflow that every AI agent must follow when working in this repository.

The goal is to maintain a clean, scalable, production-ready codebase.

---

# Primary Objectives

Every change should prioritize:

* Code correctness
* Maintainability
* Scalability
* Performance
* Security
* Readability
* Testability

Never optimize only for writing less code.

---

# Project Architecture

Follow the existing project architecture.

Do not introduce new architectural patterns unless absolutely necessary.

Maintain clear separation of concerns.

Presentation Layer
→ Handles UI/API only.

Business Logic Layer
→ Contains all application logic.

Data Layer
→ Responsible for persistence and external services.

Infrastructure Layer
→ Third-party services, storage, cloud integrations, queues, etc.

Never mix responsibilities across layers.

---

# Engineering Principles

Follow these principles:

* SOLID
* DRY
* KISS
* YAGNI
* Clean Code
* Composition over Inheritance
* Dependency Injection where appropriate

---

# Before Writing Code

Always:

1. Understand the complete feature.
2. Search for existing implementations.
3. Reuse existing utilities.
4. Avoid duplicate logic.
5. Check existing coding patterns.
6. Maintain consistency with the repository.

---

# File Modification Rules

Prefer modifying existing files.

Avoid creating new files unless:

* New module
* New service
* New feature
* New component
* New model

Never create duplicate utilities.

---

# Refactoring Guidelines

If touching existing code:

* Remove dead code
* Remove duplicated logic
* Improve naming
* Simplify complexity
* Preserve behavior
* Keep changes minimal

---

# Naming Conventions

Use descriptive names.

Good:

UserService
ReviewAnalyzer
SpeechProcessor
FeatureExtractor

Avoid:

Helper
Utils2
Temp
Manager
Thing

Variables should explain intent.

---

# Function Guidelines

Functions should:

* Have one responsibility
* Be short
* Be reusable
* Be easy to test

Avoid deeply nested logic.

Extract helper methods when needed.

---

# Error Handling

Never silently ignore exceptions.

Always:

* Validate inputs
* Return meaningful errors
* Log unexpected failures
* Fail gracefully

---

# Security

Never:

* Hardcode secrets
* Commit API keys
* Store passwords in plain text
* Disable authentication
* Skip authorization

Always:

* Validate input
* Sanitize user data
* Use environment variables
* Follow least privilege

---

# Performance

Avoid:

* N+1 queries
* Duplicate API calls
* Repeated computations
* Loading unnecessary data

Prefer:

* Caching
* Pagination
* Batch processing
* Lazy loading where appropriate

---

# Database

Keep database operations efficient.

Prefer:

* Indexed lookups
* Transactions
* Parameterized queries
* Repository pattern if applicable

Never duplicate queries across files.

---

# API Standards

APIs should:

* Validate requests
* Return proper HTTP status codes
* Use consistent response formats
* Handle errors predictably

Example:

{
"success": true,
"data": {},
"message": ""
}

---

# Logging

Log:

* Errors
* Warnings
* Important business events

Never log:

* Passwords
* Tokens
* Secrets
* Sensitive user information

---

# Testing

Every feature should be testable.

Consider:

* Unit tests
* Integration tests
* Edge cases
* Failure scenarios

Do not break existing functionality.

---

# Documentation

Update documentation whenever:

* API changes
* Architecture changes
* Environment variables change
* New dependencies are introduced

---

# Code Style

Write code that is:

* Self-explanatory
* Modular
* Consistent
* Readable

Avoid unnecessary comments.

Use comments only to explain *why*, not *what*.

---

# Dependencies

Before adding a new dependency:

* Check if an existing library already solves the problem.
* Prefer standard libraries.
* Avoid unnecessary package bloat.

---

# Git Practices

Keep commits focused.

One logical change per commit.

Avoid mixing refactoring with feature development unless required.

---

# Pull Request Checklist

Before completing work:

* Code builds successfully
* No lint errors
* No type errors
* Tests pass
* No duplicated logic
* No dead code
* Documentation updated if necessary

---

# When Implementing Features

Understand the complete requirement before coding.

Think through:

* Architecture
* Edge cases
* Security implications
* Performance impact
* Future extensibility

Do not implement partial solutions when a complete, maintainable solution is expected.

---

# When Fixing Bugs

Do not only patch the symptom.

Identify:

* Root cause
* Related code paths
* Potential regressions

Fix the underlying issue whenever possible.

---

# Communication

When making substantial changes:

1. Explain the problem.
2. Explain the proposed solution.
3. Describe trade-offs.
4. Highlight risks.
5. Summarize modified files.

Keep explanations concise and technical.

---

# Constraints

Never:

* Break existing APIs without necessity
* Introduce breaking changes silently
* Ignore existing architecture
* Duplicate business logic
* Hardcode configuration
* Leave TODOs without justification

Always leave the codebase cleaner than you found it.

---

# Success Criteria

A completed task should:

* Compile successfully
* Pass tests
* Follow repository conventions
* Be production-ready
* Be maintainable
* Be secure
* Be performant
* Be understandable by another engineer without additional explanation
