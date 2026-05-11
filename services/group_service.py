"""
Group & Pro Account service.
Handles creation, membership, invites, and message routing for collaborative groups.
"""
from __future__ import annotations

import secrets
import re
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession

from database import (
    User, Group, GroupMember, ProAccount, ProAccountMember, Message, async_session_maker
)


class GroupService:

    # ── Pro Account ──────────────────────────────────────────────────

    async def get_or_create_pro_account(self, user: User, db: AsyncSession) -> ProAccount:
        acct = await db.scalar(select(ProAccount).where(ProAccount.owner_id == user.id))
        if not acct:
            acct = ProAccount(owner_id=user.id)
            db.add(acct)
            await db.flush()
        return acct

    async def get_pro_account_for_user(self, user_id: int, db: AsyncSession) -> Optional[ProAccount]:
        """Return the Pro account a user owns OR is a member of."""
        acct = await db.scalar(select(ProAccount).where(ProAccount.owner_id == user_id))
        if acct:
            return acct
        member_row = await db.scalar(
            select(ProAccountMember).where(
                ProAccountMember.user_id == user_id,
                ProAccountMember.status == "active",
            )
        )
        if member_row:
            return await db.get(ProAccount, member_row.account_id)
        return None

    async def invite_member(
        self, inviter: User, phone_number: str, db: AsyncSession
    ) -> Dict[str, Any]:
        if not inviter.is_pro:
            return {"success": False, "message": "You need a Pro plan to invite members."}

        acct = await self.get_or_create_pro_account(inviter, db)

        # Count active/pending members (excluding owner)
        count = await db.scalar(
            select(func.count()).select_from(ProAccountMember).where(
                ProAccountMember.account_id == acct.id,
                ProAccountMember.status.in_(["pending", "active"]),
            )
        ) or 0
        if count >= acct.max_members - 1:  # owner counts as 1
            return {"success": False, "message": f"Your plan allows {acct.max_members} members total. Limit reached."}

        # Check if already invited
        existing = await db.scalar(
            select(ProAccountMember).where(
                ProAccountMember.account_id == acct.id,
                ProAccountMember.phone_number == phone_number,
            )
        )
        if existing:
            if existing.status == "active":
                return {"success": False, "message": f"{phone_number} is already an active member."}
            # Pending — refresh the token so they can re-join
            token = secrets.token_urlsafe(32)
            existing.invite_token = token
            invitee_user = await db.scalar(select(User).where(User.phone_number == phone_number))
            await db.commit()
            return {
                "success": True,
                "invite_token": token,
                "invitee_exists": invitee_user is not None,
                "invitee_name": invitee_user.name if invitee_user else None,
                "resent": True,
                "message": f"Invite resent to {phone_number}",
            }

        token = secrets.token_urlsafe(32)
        invitee_user = await db.scalar(select(User).where(User.phone_number == phone_number))

        member = ProAccountMember(
            account_id=acct.id,
            user_id=invitee_user.id if invitee_user else None,
            phone_number=phone_number,
            invited_by=inviter.id,
            invite_token=token,
            status="pending",
        )
        db.add(member)
        await db.commit()

        return {
            "success": True,
            "invite_token": token,
            "invitee_exists": invitee_user is not None,
            "invitee_name": invitee_user.name if invitee_user else None,
            "message": f"Invite sent to {phone_number}",
        }

    async def accept_invite(self, token: str, user: User, db: AsyncSession) -> Dict[str, Any]:
        member_row = await db.scalar(
            select(ProAccountMember).where(
                ProAccountMember.invite_token == token,
                ProAccountMember.status == "pending",
            )
        )
        if not member_row:
            return {"success": False, "message": "Invalid or expired invite."}

        if member_row.phone_number != user.phone_number:
            return {"success": False, "message": "This invite is for a different phone number."}

        member_row.user_id = user.id
        member_row.status = "active"
        member_row.joined_at = datetime.utcnow()
        await db.flush()

        # Auto-add to all existing groups in this Pro account
        # Fetch all groups eagerly before iterating to avoid open-cursor conflicts in async SQLAlchemy
        all_groups = (
            await db.execute(select(Group).where(Group.account_id == member_row.account_id))
        ).scalars().all()
        for group in all_groups:
            already = await db.scalar(
                select(GroupMember).where(
                    GroupMember.group_id == group.id,
                    GroupMember.user_id == user.id,
                )
            )
            if not already:
                db.add(GroupMember(group_id=group.id, user_id=user.id, role="member"))

        await db.commit()
        return {"success": True, "message": "You've joined the Pro account! ✅\n\nUse `/mygroups` to see shared groups, then `/setgroup <name>` to activate one."}

    async def cancel_invite(
        self, owner: User, phone_number: str, db: AsyncSession
    ) -> Dict[str, Any]:
        acct = await self.get_or_create_pro_account(owner, db)
        row = await db.scalar(
            select(ProAccountMember).where(
                ProAccountMember.account_id == acct.id,
                ProAccountMember.phone_number == phone_number,
                ProAccountMember.status == "pending",
            )
        )
        if not row:
            return {"success": False, "message": f"No pending invite for {phone_number}."}
        await db.delete(row)
        await db.commit()
        return {"success": True, "message": f"Invite for {phone_number} cancelled. Slot freed."}

    async def add_member_to_active_group(
        self, owner: User, phone_number: str, db: AsyncSession
    ) -> Dict[str, Any]:
        acct = await self.get_pro_account_for_user(owner.id, db)
        if not acct:
            return {"success": False, "message": "You need a Pro plan to manage groups."}

        # Verify the phone is an active member of this Pro account
        member_row = await db.scalar(
            select(ProAccountMember).where(
                ProAccountMember.account_id == acct.id,
                ProAccountMember.phone_number == phone_number,
                ProAccountMember.status == "active",
            )
        )
        if not member_row:
            return {"success": False, "message": f"{phone_number} is not an active member of your Pro account."}

        if not owner.active_group_id:
            return {"success": False, "message": "You have no active group. Use `/setgroup <name>` first."}

        group = await db.get(Group, owner.active_group_id)
        if not group:
            return {"success": False, "message": "Active group not found."}

        target_user = await db.scalar(select(User).where(User.id == member_row.user_id))
        if not target_user:
            return {"success": False, "message": f"{phone_number} has not linked their account yet."}

        existing = await db.scalar(
            select(GroupMember).where(
                GroupMember.group_id == group.id,
                GroupMember.user_id == target_user.id,
            )
        )
        if existing:
            return {"success": False, "message": f"{target_user.name} is already in *{group.name}*."}

        db.add(GroupMember(group_id=group.id, user_id=target_user.id, role="member"))
        await db.commit()
        return {"success": True, "name": target_user.name, "group_name": group.name}

    async def get_account_members(self, account_id: int, db: AsyncSession) -> List[Dict]:
        rows = await db.execute(
            select(ProAccountMember, User)
            .outerjoin(User, User.id == ProAccountMember.user_id)
            .where(ProAccountMember.account_id == account_id)
        )
        result = []
        for member_row, user in rows:
            result.append({
                "phone": member_row.phone_number,
                "name": user.name if user else "Pending",
                "status": member_row.status,
                "joined_at": member_row.joined_at.isoformat() if member_row.joined_at else None,
            })
        return result

    # ── Groups ────────────────────────────────────────────────────────

    async def create_group(
        self, creator: User, name: str, description: Optional[str], db: AsyncSession
    ) -> Dict[str, Any]:
        acct = await self.get_pro_account_for_user(creator.id, db)
        if not acct:
            return {"success": False, "message": "You need a Pro plan to create groups. Use /upgrade to learn more."}

        # Deduplicate within account
        existing = await db.scalar(
            select(Group).where(Group.account_id == acct.id, Group.name.ilike(name))
        )
        if existing:
            return {"success": False, "message": f"Group '{name}' already exists."}

        group = Group(
            account_id=acct.id,
            name=name,
            description=description,
            created_by=creator.id,
        )
        db.add(group)
        await db.flush()

        # Auto-add creator as admin
        db.add(GroupMember(group_id=group.id, user_id=creator.id, role="admin"))
        await db.commit()

        return {"success": True, "group_id": group.id, "name": group.name}

    async def add_member_to_group(
        self, group: Group, user_id: int, role: str = "member", db: AsyncSession = None
    ) -> bool:
        existing = await db.scalar(
            select(GroupMember).where(
                GroupMember.group_id == group.id, GroupMember.user_id == user_id
            )
        )
        if existing:
            return False
        db.add(GroupMember(group_id=group.id, user_id=user_id, role=role))
        await db.commit()
        return True

    async def get_user_groups(self, user_id: int, db: AsyncSession) -> List[Dict]:
        rows = await db.execute(
            select(Group, GroupMember)
            .join(GroupMember, GroupMember.group_id == Group.id)
            .where(GroupMember.user_id == user_id)
            .order_by(Group.created_at.desc())
        )
        groups = []
        for group, membership in rows:
            member_count = await db.scalar(
                select(func.count()).select_from(GroupMember).where(GroupMember.group_id == group.id)
            ) or 0
            groups.append({
                "id": group.id,
                "name": group.name,
                "description": group.description,
                "emoji": group.emoji or "👥",
                "role": membership.role,
                "member_count": member_count,
                "created_at": group.created_at.isoformat(),
            })
        return groups

    async def get_group_by_name(
        self, user_id: int, name: str, db: AsyncSession
    ) -> Optional[Group]:
        rows = await db.execute(
            select(Group)
            .join(GroupMember, GroupMember.group_id == Group.id)
            .where(GroupMember.user_id == user_id, Group.name.ilike(name))
        )
        row = rows.first()
        return row[0] if row else None

    async def get_group_by_id(self, group_id: int, user_id: int, db: AsyncSession) -> Optional[Group]:
        rows = await db.execute(
            select(Group)
            .join(GroupMember, GroupMember.group_id == Group.id)
            .where(Group.id == group_id, GroupMember.user_id == user_id)
        )
        row = rows.first()
        return row[0] if row else None

    async def get_group_members(self, group_id: int, db: AsyncSession) -> List[Dict]:
        rows = await db.execute(
            select(GroupMember, User)
            .join(User, User.id == GroupMember.user_id)
            .where(GroupMember.group_id == group_id)
        )
        return [
            {"id": u.id, "name": u.name, "phone": u.phone_number, "role": gm.role}
            for gm, u in rows
        ]

    async def get_group_messages(
        self, group_id: int, limit: int = 50, db: AsyncSession = None
    ) -> List[Dict]:
        from sqlalchemy import text
        rows = await db.execute(
            select(Message, User)
            .join(User, User.id == Message.user_id)
            .where(Message.group_id == group_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        results = []
        for msg, user in rows:
            tags = msg.tags or {}
            results.append({
                "id": msg.id,
                "content": msg.content,
                "summary": msg.summary or "",
                "message_type": msg.message_type.value,
                "tags": tags,
                "sender_name": user.name,
                "sender_id": user.id,
                "assigned_to_user_id": msg.assigned_to_user_id,
                "created_at": msg.created_at.isoformat(),
                "media_url": msg.media_url,
            })
        return results

    # ── Active group management ───────────────────────────────────────

    async def set_active_group(self, user_id: int, group_id: Optional[int], db: AsyncSession):
        await db.execute(
            update(User).where(User.id == user_id).values(active_group_id=group_id)
        )
        await db.commit()

    # ── @mention parsing ──────────────────────────────────────────────

    def parse_mention(self, content: str, members: List[Dict]) -> tuple[Optional[int], str]:
        """
        Detect @name prefix and return (assigned_user_id, cleaned_content).
        Matches case-insensitively against first names of group members.
        """
        m = re.match(r"^@(\w+)[:\s]+(.+)$", content, re.DOTALL | re.IGNORECASE)
        if not m:
            return None, content
        mention_name = m.group(1).lower()
        rest = m.group(2).strip()
        for member in members:
            if member["name"].split()[0].lower() == mention_name:
                return member["id"], rest
        return None, content

    # ── Bot response helpers ──────────────────────────────────────────

    def format_groups_list(self, groups: List[Dict]) -> str:
        if not groups:
            return (
                "You have no groups yet.\n\n"
                "Create one: `/creategroup Goa Trip`\n"
                "Or invite members: `/invite +91XXXXXXXXXX`"
            )
        lines = ["👥 *Your Groups*\n"]
        for g in groups:
            lines.append(
                f"{g['emoji']} *{g['name']}* — {g['member_count']} members\n"
                f"  `/setgroup {g['name']}`"
            )
        lines.append("\n_Active group messages go to that group's shared space._")
        return "\n".join(lines)

    def format_group_members(self, group_name: str, members: List[Dict]) -> str:
        if not members:
            return f"*{group_name}* has no members yet."
        lines = [f"👥 *{group_name}* members:\n"]
        for m in members:
            role_badge = "👑 " if m["role"] == "admin" else "• "
            lines.append(f"{role_badge}*{m['name']}* — @{m['name'].split()[0].lower()}")
        lines.append(
            "\n_Assign tasks: `@name: call Ashu about the hotel`_"
        )
        return "\n".join(lines)


group_service = GroupService()
