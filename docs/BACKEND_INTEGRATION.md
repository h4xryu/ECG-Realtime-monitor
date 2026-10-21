# Backend integration (signup/login + ECG upload)

Before this change, neither this Android app nor the companion
`Live-ECG-Monitor-WebService` web project had any backend: no networking code here,
no API routes there. This adds a minimal account system and an ECG-upload endpoint to
the web project, and wires this app up as its client.

## What's on the server side

In the `Live-ECG-Monitor-WebService` Next.js project (not this repo):
- `app/api/auth/signup`, `app/api/auth/login`, `app/api/auth/me` — JWT-based auth.
- `app/api/ecg/upload`, `app/api/ecg/records` — authenticated ECG record storage.
- File-backed storage under `data/` (no external database needed for a single-instance
  dev/demo server). Full request/response shapes are in that project's
  `docs/BACKEND_API.md`.

Run it with `pnpm install && pnpm dev` (copy `.env.local.example` to `.env.local` and
set a real `JWT_SECRET` first).

## What's on this (Android) side

- `data/network/ApiConfig.kt` — the backend base URL, stored in SharedPreferences and
  editable from Settings (defaults to `http://10.0.2.2:3000/api/`, the emulator's alias
  for the host machine).
- `data/network/ApiClient.kt` — a small OkHttp + JSON wrapper (`ApiResult<T>` success/
  failure), used synchronously; callers dispatch it off the main thread themselves.
- `data/network/SessionManager.kt` — stores the JWT + username after login/signup.
- `data/network/AuthRepository.kt` — `signUp`, `login`, `validateSession`, `logout`.
- `data/network/EcgUploadRepository.kt` — uploads a `DiagnosisRecord`'s filtered signal
  + metadata to `/ecg/upload`.
- `ui/auth/LoginActivity.kt` / `ui/auth/SignupActivity.kt` — new account screens.
- Settings tab — "클라우드 계정" section: shows login state, lets you edit the server
  URL, log in/sign up/out.
- Record detail screen — "서버로 업로드" button; prompts login first if needed, then
  uploads that record.

## Trying it end to end

1. Start the web backend (`pnpm dev`, default port 3000).
2. Run this app on an emulator — the default base URL (`http://10.0.2.2:3000/api/`)
   already points at it. For a physical device, run
   `adb reverse tcp:3000 tcp:3000` and set the server URL in Settings to
   `http://localhost:3000/api/` instead.
3. Settings → 클라우드 계정 → 회원가입, create an account.
4. Open any saved recording → "서버로 업로드".

## Known limitations / next steps

- The app still works fully offline — login is optional, matching the existing
  on-device-first design. Nothing blocks the monitor or local recording if the user
  never signs in.
- The file-backed server store is single-process only; swap `lib/db.ts` on the server
  for a real database before running more than one instance.
- The JWT is stored in plain `SharedPreferences`, not `EncryptedSharedPreferences` —
  fine for this MVP (the token only exposes this one user's own data on a dev server),
  but should be hardened before this touches real patient data.
- Uploads send the whole recording in one request (capped at 2,000,000 samples ≈ 2.2h
  at 250Hz, matching the server's limit). A very long continuous recording would need
  to be chunked across multiple uploads instead of raising that cap.
- `network_security_config.xml` only allows plaintext HTTP to `10.0.2.2` and
  `localhost`, for local development. Once the backend is served over HTTPS, delete
  that file and the manifest's `android:networkSecurityConfig` reference.
