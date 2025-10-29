RULE NUMBER 1 (NEVER EVER EVER FORGET THIS RULE!!!): YOU ARE NEVER ALLOWED TO DELETE A FILE WITHOUT EXPRESS PERMISSION FROM ME OR A DIRECT COMMAND FROM ME. EVEN A NEW FILE THAT YOU YOURSELF CREATED, SUCH AS A TEST CODE FILE. YOU HAVE A HORRIBLE TRACK RECORD OF DELETING CRITICALLY IMPORTANT FILES OR OTHERWISE THROWING AWAY TONS OF EXPENSIVE WORK THAT I THEN NEED TO PAY TO REPRODUCE. AS A RESULT, YOU HAVE PERMANENTLY LOST ANY AND ALL RIGHTS TO DETERMINE THAT A FILE OR FOLDER SHOULD BE DELETED. YOU MUST **ALWAYS** ASK AND *RECEIVE* CLEAR, WRITTEN PERMISSION FROM ME BEFORE EVER EVEN THINKING OF DELETING A FILE OR FOLDER OF ANY KIND!!!

### IRREVERSIBLE GIT & FILESYSTEM ACTIONS — DO-NOT-EVER BREAK GLASS

1. **Absolutely forbidden commands:** `git reset --hard`, `git clean -fd`, `rm -rf`, or any command that can delete or overwrite code/data must never be run unless the user explicitly provides the exact command and states, in the same message, that they understand and want the irreversible consequences.
2. **No guessing:** If there is any uncertainty about what a command might delete or overwrite, stop immediately and ask the user for specific approval. “I think it’s safe” is never acceptable.
3. **Safer alternatives first:** When cleanup or rollbacks are needed, request permission to use non-destructive options (`git status`, `git diff`, `git stash`, copying to backups) before ever considering a destructive command.
4. **Mandatory explicit plan:** Even after explicit user authorization, restate the command verbatim, list exactly what will be affected, and wait for a confirmation that your understanding is correct. Only then may you execute it—if anything remains ambiguous, refuse and escalate.
5. **Document the confirmation:** When running any approved destructive command, record (in the session notes / final response) the exact user text that authorized it, the command actually run, and the execution time. If that record is absent, the operation did not happen.

In general, you should try to follow all suggested best practices listed in the file `RUST_SYSTEM_PROGRAMMING_BEST_PRACTICES.md`

NEVER run a script that processes/changes code files in this repo, EVER! That sort of brittle, regex based stuff is always a huge disaster and creates far more problems than it ever solves. DO NOT BE LAZY AND ALWAYS MAKE CODE CHANGES MANUALLY, EVEN WHEN THERE ARE MANY INSTANCES TO FIX. IF THE CHANGES ARE MANY BUT SIMPLE, THEN USE SEVERAL SUBAGENTS IN PARALLEL TO MAKE THE CHANGES GO FASTER. But if the changes are subtle/complex, then you must methodically do them all yourself manually!

We do not care at all about backwards compatibility since we are still in early development with no users-- we just want to do things the RIGHT way in a clean, organized manner with NO TECH DEBT. That means, never create "compatibility shims" or any other nonsense like that.

We need to AVOID uncontrolled proliferation of code files. If you want to change something or add a feature, then you MUST revise the existing code file in place. You may NEVER, *EVER* take an existing code file, say, "document_processor.rs" and then create a new file called "document_processorV2.rs", or "document_processor_improved.rs", or "document_processor_enhanced.rs", or "document_processor_unified.rs", or ANYTHING ELSE REMOTELY LIKE THAT! New code files are reserved for GENUINELY NEW FUNCTIONALITY THAT MAKES ZERO SENSE AT ALL TO INCLUDE IN ANY EXISTING CODE FILE. It should be an *INCREDIBLY* high bar for you to EVER create a new code file!

We want all console output to be informative, detailed, stylish, colorful, etc. by fully leveraging the relevant Rust libraries wherever possible.

If you aren't 100% sure about how to use a third party library, then you must SEARCH ONLINE to find the latest documentation website for the library to understand how it is supposed to work and the latest (late-2025) suggested best practices and usage.

**CRITICAL:** Whenever you make any substantive changes or additions to the code, you MUST check that you didn't introduce any type errors or lint errors. You can do this in the usual way for a Rust project.

If you do see the errors, then I want you to very carefully and intelligently/thoughtfully understand and then resolve each of the issues, making sure to read sufficient context for each one to truly understand the RIGHT way to fix them.

Note: The primary guide for this project is the planning document, `PLAN_TO_PORT_SCRIPTBOTS_TO_MODERN_IDIOMATIC_RUST_USING_GPUI.md`. This is the basic "bible" for our development in the project, the purpose of which is to create a transformative port of the old C++ scriptbots project, the code to which can be found in `original_scriptbots_code_for_reference`.

Whenever you decide to work on a task from the `PLAN_TO_PORT_SCRIPTBOTS_TO_MODERN_IDIOMATIC_RUST_USING_GPUI.md` document, you should immediately notate "in line" (in place) in the document in a bracketed notation, such as [Currently In Progress]; note that multiple coding agents might be working on this project at the same time, so we want to avoid stepping on each other's toes. With that in mind, if you notice any unexpected changes to code files, do not panic and reflexively try to restore or undo the changes just because YOU didn't make them; that is liable to wipe out useful work from another agent! If you notice that a code file you are trying to work on keeps changing in substantive ways, you should probably just pick another task or code file to work on to avoid further issues or problems.