# Lessons Learned

Capture failure modes, detection signals, and prevention rules.

---

## 2026-02-07: Always use git workflow before making changes

**Failure mode**: Made changes directly on `main` without creating a branch first.

**Detection signal**: Asked "did we create a branch?" after changes were already made.

**Prevention rule**: Before making any code changes, always follow this workflow:
1. Create a feature branch (`git checkout -b <branch-name>`)
2. Make changes
3. Commit with clear message
4. Push to remote
5. Create PR
6. Merge after review

Never make changes directly on `main`.

---

## 2026-02-08: Link PRs to issues with "Fixes #X"

**Failure mode**: PR #46 implemented issue #27 but didn't reference it, so the issue stayed open after merge.

**Detection signal**: Had to manually close issue #27 after noticing it was still open.

**Prevention rule**: Always include `Fixes #X` or `Closes #X` in PR description to auto-close issues on merge.

---

## 2026-02-08: GitHub math rendering requires special formatting

**Failure mode**: LaTeX math in markdown didn't render correctly on GitHub. Subscripts like `$\mathbf{q}_{\min}$` displayed as `$\mathbf{q}{\min}$` with missing underscores.

**Detection signal**: User reported math looked broken when viewing on GitHub.

**Prevention rules**:
1. **Display math**: Use fenced code blocks with `math` language, not `$$`:
   ```
   ```math
   \mathbf{H} = \mathbf{J}^T \mathbf{W} \mathbf{J}
   ```
   ```
2. **Inline math**: Escape underscores with backslash: `$\mathbf{q}\_{\min}$` not `$\mathbf{q}_{\min}$`
3. GitHub's markdown parser processes `_` as italic markers before LaTeX renders, so unescaped underscores in inline math get stripped.

---

## 2026-02-08: Review code for attribute name consistency

**Failure mode**: `Arm.pickup()` and `Arm.place()` used `self._robot` but the attribute was stored as `self.robot`. Would have caused `AttributeError` at runtime.

**Detection signal**: Code review during pre-1.0 cleanup caught the inconsistency.

**Prevention rule**: When referencing instance attributes, verify the exact name used in `__init__`. Python convention: `self._name` = private, `self.name` = public. Don't mix them for the same attribute.
