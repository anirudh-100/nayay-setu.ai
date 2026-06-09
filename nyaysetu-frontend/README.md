# NyaySetu Frontend

A friendly, mobile-first chat UI for the NyaySetu legal assistant.

- **Next.js 14** (App Router) + **TypeScript** + **Tailwind**
- **Bilingual** — English / हिन्दी toggle (UI strings; the engine answers in English today)
- Renders the engine's trust features: **confidence badge**, **clickable citations** with
  current/repealed chips, the **IPC→BNS current-law note**, **unverified-citation warning**,
  the **action** step, and the **legal-aid escalation** banner.

## Run

```bash
cd nyaysetu-frontend
cp .env.local.example .env.local      # point NEXT_PUBLIC_API_URL at the backend
npm install
npm run dev                           # http://localhost:3000
```

The backend (`nyaysetu-backend`) must be running and reachable at `NEXT_PUBLIC_API_URL`
(default `http://127.0.0.1:8000`).

> **CORS:** the FastAPI backend needs to allow the frontend origin. If requests fail with a
> network error, add `CORSMiddleware` for `http://localhost:3000` in `app/main.py`.

## Structure

```
src/
├── app/
│   ├── layout.tsx        # root layout, fonts, i18n provider
│   ├── page.tsx          # renders <ChatApp/>
│   └── globals.css       # theme tokens (calm teal palette)
├── components/
│   ├── ChatApp.tsx       # stateful container (conversation, API calls, reset)
│   ├── Header.tsx        # brand + language toggle + new-question
│   ├── Composer.tsx      # auto-growing input, Enter-to-send, example chips
│   ├── MessageBubble.tsx # user bubble / thinking / error / answer
│   ├── AnswerCard.tsx    # the full structured response
│   ├── CitationCard.tsx  # one source (icon, status chip, snippet, link)
│   ├── ConfidenceBadge.tsx
│   └── LanguageToggle.tsx
├── lib/
│   ├── api.ts            # typed POST /ask client
│   ├── types.ts          # mirrors the backend response contract
│   ├── i18n.tsx          # language context + t()
│   └── cn.ts             # className merge helper
└── i18n/
    ├── en.json
    └── hi.json
```

## Notes

- Branding is config-driven (`NEXT_PUBLIC_APP_NAME` / `NEXT_PUBLIC_APP_TAGLINE`).
- Respects `prefers-reduced-motion`; accessible labels on controls.
- Voice input and a fuller landing page are planned for a later phase.
