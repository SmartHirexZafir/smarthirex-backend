# app/utils/emailer.py
"""
SMTP email helper for SmartHirex.

Env vars:
- SMTP_HOST, SMTP_PORT
- SMTP_USER, SMTP_PASS
- FROM_NAME (optional, defaults to "SmartHirex Team")
- SMTP_SSL (optional: "true"/"false", default false → STARTTLS)

Provides:
- send_email(...)                 -> sync
- async_send_email(...)           -> async wrapper (for FastAPI)
- render_invite_html(...)         -> default HTML template for test invites
- render_interview_invite_html(...)  -> default HTML for interview invites
- send_interview_invite(...)         -> thin wrapper around send_email for interviews
- async_send_interview_invite(...)   -> async wrapper for interviews

# ✅ NEW (non-breaking) for account verification:
- render_verify_email_html(verify_url)     -> HTML template for email verification
- send_verification_email(to, verify_url)  -> sends verification email
- async_send_verification_email(...)       -> async wrapper
"""

from __future__ import annotations

import os
import ssl
import smtplib
import mimetypes
import re
from email.message import EmailMessage
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union, Dict, Any
import asyncio
import html as html_unescape_mod


def _env(name: str, default: Optional[str] = None) -> str:
    val = os.getenv(name, default)
    if val is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


def _get_smtp_config() -> dict:
    return {
        "host": _env("SMTP_HOST"),
        "port": int(os.getenv("SMTP_PORT", "587")),
        "user": _env("SMTP_USER"),
        "password": _env("SMTP_PASS"),
        "from_name": os.getenv("FROM_NAME", "SmartHirex Team"),
        "use_ssl": os.getenv("SMTP_SSL", "false").lower() in {"1", "true", "yes"},
    }


Address = Union[str, Sequence[str]]


def _normalize_addresses(addr: Optional[Address]) -> list[str]:
    if addr is None:
        return []
    if isinstance(addr, str):
        return [addr]
    return [a for a in addr if a]


def _strip_tags(s: str) -> str:
    """Very small tag stripper for fallback."""
    return re.sub(r"<[^>]+>", "", s)


def _html_to_text(html: str) -> str:
    """
    Lightweight HTML -> plain text converter for email text part.
    - <br> -> newline
    - </p> -> blank line
    - <a href="URL">label</a> -> "label (URL)"
    - remove remaining tags
    - unescape HTML entities
    """
    if not isinstance(html, str):
        return ""

    # Normalize newlines for common block elements
    # <br> → \n
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)

    # paragraphs → blank line
    html = re.sub(r"</p\s*>", "\n\n", html, flags=re.I)

    # links → label (href)
    def _a_sub(m: re.Match) -> str:
        href = (m.group(1) or "").strip()
        inner = (m.group(2) or "").strip()
        # if no label, just return href
        if inner:
            return f"{inner} ({href})" if href else inner
        return href

    html = re.sub(
        r'<a[^>]*?href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
        _a_sub,
        html,
        flags=re.I | re.S,
    )

    # Remove other tags
    html = _strip_tags(html)

    # Collapse excessive whitespace
    html = re.sub(r"[ \t]+\n", "\n", html)
    html = re.sub(r"\n{3,}", "\n\n", html)

    # Unescape entities (&amp; → &)
    html = html_unescape_mod.unescape(html)

    # Trim
    return html.strip()


def build_message(
    *,
    to: Address,
    subject: str,
    html: str,
    text: Optional[str] = None,
    from_name: Optional[str] = None,
    cc: Optional[Address] = None,
    bcc: Optional[Address] = None,
    reply_to: Optional[str] = None,
    attachments: Optional[Iterable[Union[str, Path]]] = None,
) -> EmailMessage:
    cfg = _get_smtp_config()
    from_addr = cfg["user"]
    from_name = from_name or cfg["from_name"]

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"{from_name} <{from_addr}>"

    to_list = _normalize_addresses(to)
    if not to_list:
        raise ValueError("'to' must contain at least one address.")
    msg["To"] = ", ".join(to_list)

    cc_list = _normalize_addresses(cc)
    bcc_list = _normalize_addresses(bcc)
    if cc_list:
        msg["Cc"] = ", ".join(cc_list)
    if reply_to:
        msg["Reply-To"] = reply_to

    # Build text part (fallback to HTML→text)
    text = text or _html_to_text(html) or "Your email client does not support HTML."

    msg.set_content(text)
    msg.add_alternative(html, subtype="html")

    for att in attachments or []:
        p = Path(att)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Attachment not found: {p}")
        ctype, encoding = mimetypes.guess_type(p.name)
        if ctype is None or encoding is not None:
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)
        with p.open("rb") as fp:
            msg.add_attachment(fp.read(), maintype=maintype, subtype=subtype, filename=p.name)

    if bcc_list:
        # keep Bcc out of headers; track via X-Bcc then remove before send
        msg["X-Bcc"] = ", ".join(bcc_list)

    return msg


def send_email(
    *,
    to: Address,
    subject: str,
    html: str,
    text: Optional[str] = None,
    from_name: Optional[str] = None,
    cc: Optional[Address] = None,
    bcc: Optional[Address] = None,
    reply_to: Optional[str] = None,
    attachments: Optional[Iterable[Union[str, Path]]] = None,
) -> None:
    cfg = _get_smtp_config()
    msg = build_message(
        to=to, subject=subject, html=html, text=text,
        from_name=from_name, cc=cc, bcc=bcc, reply_to=reply_to,
        attachments=attachments,
    )

    rcpts: list[str] = []
    for key in ("To", "Cc"):
        if key in msg:
            rcpts.extend([a.strip() for a in msg[key].split(",") if a.strip()])
    if "X-Bcc" in msg:
        rcpts.extend([a.strip() for a in msg["X-Bcc"].split(",") if a.strip()])
        del msg["X-Bcc"]

    context = ssl.create_default_context()
    if cfg["use_ssl"]:
        with smtplib.SMTP_SSL(cfg["host"], cfg["port"], context=context) as s:
            s.login(cfg["user"], cfg["password"])
            s.send_message(msg, to_addrs=rcpts)
    else:
        with smtplib.SMTP(cfg["host"], cfg["port"]) as s:
            s.ehlo()
            s.starttls(context=context)
            s.ehlo()
            s.login(cfg["user"], cfg["password"])
            s.send_message(msg, to_addrs=rcpts)


async def async_send_email(**kwargs) -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: send_email(**kwargs))


def render_invite_html(
    *,
    candidate_name: str,
    role: str,
    test_link: str,
    scheduled_datetime: Optional[Any] = None,
    duration_minutes: int = 60,
) -> str:
    scheduled_info = ""
    if scheduled_datetime:
        if isinstance(scheduled_datetime, str):
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(scheduled_datetime.replace('Z', '+00:00'))
                scheduled_info = f"""
      <p><strong>Scheduled Time:</strong> {dt.strftime('%A, %B %d, %Y at %I:%M %p UTC')}</p>
      <p style="color:#dc2626;font-weight:500;">⚠️ The test will only be available at the scheduled time. Please wait until then to access it.</p>
      """
            except Exception:
                scheduled_info = f"<p><strong>Scheduled Time:</strong> {scheduled_datetime}</p>"
        else:
            scheduled_info = f"""
      <p><strong>Scheduled Time:</strong> {scheduled_datetime.strftime('%A, %B %d, %Y at %I:%M %p UTC')}</p>
      <p style="color:#dc2626;font-weight:500;">⚠️ The test will only be available at the scheduled time. Please wait until then to access it.</p>
      """
    
    duration_info = f"<p><strong>Test Duration:</strong> {duration_minutes} minutes</p>"
    if duration_minutes < 60:
        duration_info += f'<p style="color:#dc2626;">⚠️ The test will automatically submit after {duration_minutes} minutes.</p>'
    
    return f"""
    <div style="font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; font-size:14px; line-height:1.6;">
      <p>Hi <strong>{candidate_name}</strong>,</p>
      <p>You're invited to take a short <strong>{role}</strong> assessment on SmartHirex.</p>
      {scheduled_info}
      {duration_info}
      <p>
        When you're ready, click the link below to begin:<br/>
        <a href="{test_link}" style="display:inline-block;padding:10px 16px;text-decoration:none;border-radius:6px;border:1px solid #e5e7eb;background:#f9fafb;">Start your assessment</a>
      </p>
      <p>If the button doesn't work, copy and paste this URL into your browser:<br/>
        <span style="color:#6b7280;">{test_link}</span>
      </p>
      <p style="color:#6b7280;">Best of luck!<br/>SmartHirex Team</p>
    </div>
    """.strip()


# -----------------------------------------------------------------------------
# ✅ NEW: Account verification helpers (non-breaking additions)
# -----------------------------------------------------------------------------

def render_verify_email_html(*, verify_url: str) -> str:
    """Minimal branded HTML for account verification."""
    brand = os.getenv("FROM_NAME", "SmartHirex Team")
    return f"""
<!doctype html>
<html>
  <body style="margin:0;padding:0;background:#f6f7fb;">
    <table width="100%" cellpadding="0" cellspacing="0" border="0" style="background:#f6f7fb;">
      <tr>
        <td align="center" style="padding:24px;">
          <table width="600" cellpadding="0" cellspacing="0" border="0" style="max-width:600px;background:#ffffff;border-radius:14px;box-shadow:0 2px 10px rgba(16,24,40,0.06);overflow:hidden;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,'Noto Sans',sans-serif;">
            <tr>
              <td style="background:linear-gradient(135deg,#2563eb,#1d4ed8);padding:20px 24px;color:#fff;">
                <div style="font-size:18px;font-weight:700;">{brand}</div>
                <div style="font-size:12px;opacity:.9;margin-top:4px;">Verify your email</div>
              </td>
            </tr>
            <tr>
              <td style="padding:24px;">
                <p style="margin:0 0 12px 0;font-size:16px;color:#111827;">Welcome!</p>
                <p style="margin:0 0 16px 0;font-size:14px;color:#374151;line-height:1.6;">
                  Please confirm your email address to activate your account.
                </p>
                <table cellpadding="0" cellspacing="0" border="0" align="center" style="margin:8px auto 8px auto;">
                  <tr>
                    <td align="center" bgcolor="#2563eb" style="border-radius:10px;">
                      <a href="{verify_url}" target="_blank" style="display:inline-block;padding:12px 22px;font-size:14px;font-weight:600;color:#ffffff;background:#2563eb;border-radius:10px;text-decoration:none;">
                        Verify my email
                      </a>
                    </td>
                  </tr>
                </table>
                <p style="margin:14px 0 0 0;font-size:12px;color:#6b7280;line-height:1.6;">
                  If the button doesn’t work, copy and paste this link into your browser:<br />
                  <a href="{verify_url}" target="_blank" style="color:#2563eb;word-break:break-all;">{verify_url}</a>
                </p>
              </td>
            </tr>
            <tr><td style="padding:0 24px 20px 24px;"><p style="margin:0;font-size:11px;color:#9ca3af;">© {os.getenv('EMAIL_COPYRIGHT_YEAR', '')} {brand}.</p></td></tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>
""".strip()


def send_verification_email(to: str, verify_url: str) -> None:
    """Send account verification email using the shared SMTP helper."""
    subject = "Verify your email"
    html = render_verify_email_html(verify_url=verify_url)
    send_email(to=to, subject=subject, html=html)


async def async_send_verification_email(*, to: str, verify_url: str) -> None:
    await async_send_email(to=to, subject="Verify your email", html=render_verify_email_html(verify_url=verify_url))


# -----------------------------------------------------------------------------
# ✅ Interview invite helpers (existing, kept as non-breaking additions)
# -----------------------------------------------------------------------------

def render_interview_invite_html(
    *,
    candidate_name: str | None,
    title: str,
    local_datetime_str: str,
    timezone_name: str,
    meeting_url: str,
    duration_mins: int,
    notes: str | None = None,
    company_name: str | None = None,
) -> str:
    """
    Render a clean HTML email for interview invitations.
    Minimal dependencies; no template engine required.
    """
    who = candidate_name or "Candidate"
    company = company_name or os.getenv("FROM_NAME", "SmartHirex Team")
    notes_block = f"<tr><td style='padding:4px 0; color:#6b7280; vertical-align:top;'>Notes</td><td style='padding:4px 0;'>{notes}</td></tr>" if notes else ""
    return f"""\
<!doctype html>
<html>
  <body style="margin:0;padding:0;background:#f6f7fb;">
    <table width="100%" cellpadding="0" cellspacing="0" border="0" style="background:#f6f7fb;">
      <tr>
        <td align="center" style="padding:24px;">
          <table width="600" cellpadding="0" cellspacing="0" border="0" style="max-width:600px;background:#ffffff;border-radius:14px;box-shadow:0 2px 10px rgba(16,24,40,0.06);overflow:hidden;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,'Noto Sans',sans-serif;">
            <tr>
              <td style="background:linear-gradient(135deg,#6366f1,#8b5cf6);padding:20px 24px;color:#fff;">
                <div style="font-size:18px;font-weight:700;">{company}</div>
                <div style="font-size:12px;opacity:.9;margin-top:4px;">Interview Invitation</div>
              </td>
            </tr>
            <tr>
              <td style="padding:24px;">
                <p style="margin:0 0 12px 0;font-size:16px;color:#111827;">Hi <strong>{who}</strong>,</p>
                <p style="margin:0 0 16px 0;font-size:14px;color:#374151;line-height:1.6;">
                  You're invited to an interview.
                </p>
                <table width="100%" cellpadding="0" cellspacing="0" border="0" style="background:#f9fafb;border:1px solid #eef2f7;border-radius:10px;">
                  <tr>
                    <td style="padding:16px 18px;">
                      <table width="100%" cellpadding="0" cellspacing="0" border="0" style="font-size:14px;color:#111827;">
                        <tr><td style="padding:4px 0;width:120px;color:#6b7280;">Title</td><td style="padding:4px 0;font-weight:600;">{title}</td></tr>
                        <tr><td style="padding:4px 0;color:#6b7280;">When</td><td style="padding:4px 0;">{local_datetime_str} <span style="color:#6b7280;">({timezone_name})</span></td></tr>
                        <tr><td style="padding:4px 0;color:#6b7280;">Duration</td><td style="padding:4px 0;">{duration_mins} minutes</td></tr>
                        {notes_block}
                      </table>
                    </td>
                  </tr>
                </table>
                <table cellpadding="0" cellspacing="0" border="0" align="center" style="margin:22px auto 8px auto;">
                  <tr>
                    <td align="center" bgcolor="#4f46e5" style="border-radius:10px;">
                      <a href="{meeting_url}" target="_blank" style="display:inline-block;padding:12px 22px;font-size:14px;font-weight:600;color:#ffffff;background:#4f46e5;border-radius:10px;text-decoration:none;">
                        Join Interview
                      </a>
                    </td>
                  </tr>
                </table>
                <p style="margin:14px 0 0 0;font-size:12px;color:#6b7280;line-height:1.6;">
                  If the button doesn’t work, copy and paste this link into your browser:<br />
                  <a href="{meeting_url}" target="_blank" style="color:#4f46e5;word-break:break-all;">{meeting_url}</a>
                </p>
                <hr style="border:none;border-top:1px solid #eef2f7;margin:20px 0;" />
                <p style="margin:0;font-size:12px;color:#6b7280;">We look forward to speaking with you!</p>
              </td>
            </tr>
            <tr><td style="padding:0 24px 20px 24px;"><p style="margin:0;font-size:11px;color:#9ca3af;">© {os.getenv('EMAIL_COPYRIGHT_YEAR', '')} {company}. This message was intended for {who}.</p></td></tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>""".strip()


def send_interview_invite(
    *,
    to: str,
    subject: str,
    html: str,
    meta: Optional[Dict[str, Any]] = None,  # kept for API compatibility; ignored here
) -> None:
    """
    Thin, backward-compatible wrapper used by interview routes/services.
    Matches the expected signature:
        send_interview_invite(to=..., subject=..., html=..., meta={...})
    """
    # You can attach ICS/metadata here later if desired.
    send_email(to=to, subject=subject, html=html)


async def async_send_interview_invite(
    *,
    to: str,
    subject: str,
    html: str,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    await async_send_email(to=to, subject=subject, html=html)
