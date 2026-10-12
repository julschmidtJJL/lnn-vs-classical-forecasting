#!/usr/bin/env bash
# Watch a detached docker run and email when it exits.
#
# Why this exists: long unattended grid runs on a cloud VM (see CLOUD_RUN.md).
# This is NOT part of the experiment image — it runs on the host box, tails the
# container's exit state, and emails DONE/FAILED so you don't babysit the run.
#
# Channel: Gmail SMTP via curl on port 587 + STARTTLS.
#   Hetzner BLOCKS port 465 (smtps), so 587 STARTTLS is mandatory there.
# Creds: read from notify.env next to this script (gitignored). Never committed.
#   Copy notify.env.example -> notify.env and fill in. The Gmail value must be a
#   16-char *app password* (Google Account -> Security -> App passwords), not the
#   normal login password. App passwords are disposable; regenerate as needed.
#
# Usage on the box:
#   ./notify_watch.sh --test          # send a test email now, then exit
#   setsid nohup ./notify_watch.sh grid > /tmp/notify.log 2>&1 &   # watch container "grid"
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$DIR/notify.env"
POLL="${POLL_SECONDS:-60}"

send_mail() {
  local subject="$1" body="$2" tmp
  tmp="$(mktemp)"
  {
    printf 'From: %s\r\n' "$GMAIL_USER"
    printf 'To: %s\r\n' "$MAIL_TO"
    printf 'Subject: %s\r\n' "$subject"
    printf 'MIME-Version: 1.0\r\n'
    printf 'Content-Type: text/plain; charset=UTF-8\r\n'
    printf '\r\n'
    printf '%s\r\n' "$body"
  } > "$tmp"
  curl --silent --show-error --ssl-reqd \
    --url 'smtp://smtp.gmail.com:587' \
    --user "$GMAIL_USER:$GMAIL_APP_PASSWORD" \
    --mail-from "$GMAIL_USER" \
    --mail-rcpt "$MAIL_TO" \
    --upload-file "$tmp"
  rm -f "$tmp"
}

if [ "${1:-}" = "--test" ]; then
  send_mail "[Thesis] notifier test ✅" "If you got this, the box can send mail. Sent from $(hostname) at $(date -u)."
  echo "test mail sent to $MAIL_TO"
  exit 0
fi

CONTAINER="${1:-grid}"

# Wait for the container to appear (in case the watcher starts first).
while ! docker inspect "$CONTAINER" >/dev/null 2>&1; do sleep "$POLL"; done
# Poll until it stops running.
while [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = "true" ]; do
  sleep "$POLL"
done

EXIT_CODE="$(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER" 2>/dev/null || echo '?')"
TAIL="$(docker logs --tail 40 "$CONTAINER" 2>&1 || true)"
HOST="$(hostname)"

if [ "$EXIT_CODE" = "0" ]; then
  send_mail "[Thesis] Full grid DONE (exit 0)" "✅ Container $CONTAINER on $HOST exited cleanly.

Confirm experiment_summaries rows in Supabase, then DELETE the VM to stop billing.

--- last 40 log lines ---
$TAIL"
else
  send_mail "[Thesis] Full grid FAILED (exit $EXIT_CODE)" "❌ Container $CONTAINER on $HOST exited $EXIT_CODE.

Resume with the SAME RUN_ID after fixing (completed ticker/seed pairs are skipped).

--- last 40 log lines ---
$TAIL"
fi
echo "notified: $CONTAINER exit=$EXIT_CODE"
