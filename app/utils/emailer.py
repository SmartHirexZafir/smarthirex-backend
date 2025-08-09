# app/utils/emailer.py
"""
SMTP email helper for SmartHirex.

Env vars:
- SMTP_HOST, SMTP_PORT
- SMTP_USER, SMTP_PASS
- FROM_NAME (optional, defaults to "SmartHirex Team")
- SMTP_SSL (optional: "true"/"false", default false → STARTTLS)

Provides:
- send_email(...)       -> sync
- async_send_email(...) -> async wrapper (for FastAPI)
- render_invite_html(...) -> default HTML template for test invites
"""

from __future__ import annotations

import os
import ssl
import smtplib
import mimetypes
from email.message import EmailMessage
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union
import asyncio


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

    if not text:
        text = "Your email client does not support HTML. Please view this message in an HTML-capable client."
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


def render_invite_html(*, candidate_name: str, role: str, test_link: str) -> str:
    return f"""
    <div style="font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; font-size:14px; line-height:1.6;">
      <p>Hi <strong>{candidate_name}</strong>,</p>
      <p>You’re invited to take a short <strong>{role}</strong> assessment on SmartHirex.</p>
      <p>
        When you’re ready, click the link below to begin:<br/>
        <a href="{test_link}" style="display:inline-block;padding:10px 16px;text-decoration:none;border-radius:6px;border:1px solid #e5e7eb;">Start your assessment</a>
      </p>
      <p>If the button doesn’t work, copy and paste this URL into your browser:<br/>
        <span style="color:#6b7280;">{test_link}</span>
      </p>
      <p style="color:#6b7280;">Best of luck!<br/>SmartHirex Team</p>
    </div>
    """.strip()
