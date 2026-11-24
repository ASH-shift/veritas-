"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { LogOut, Moon, ShieldCheck, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/hooks/use-auth";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/demo", label: "Demo", auth: false },
  { href: "/chat", label: "Chat", auth: true },
  { href: "/documents", label: "Documents", auth: true },
  { href: "/evals", label: "Evals", auth: false },
];

function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  return (
    <Button
      variant="ghost"
      size="icon"
      aria-label="Toggle dark mode"
      onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
    >
      <Sun className="hidden dark:block" />
      <Moon className="dark:hidden" />
    </Button>
  );
}

export function AppHeader() {
  const pathname = usePathname();
  const router = useRouter();
  const { token, ready, logout } = useAuth();
  const loggedIn = ready && !!token;

  return (
    <header className="sticky top-0 z-40 border-b bg-background/80 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-6xl items-center gap-2 px-4">
        <Link href="/" className="mr-2 flex items-center gap-2 font-semibold">
          <ShieldCheck className="size-5 text-primary" />
          <span>Veritas</span>
        </Link>
        <nav className="flex flex-1 items-center gap-1 overflow-x-auto">
          {NAV.filter((n) => !n.auth || loggedIn).map((n) => (
            <Link
              key={n.href}
              href={n.href}
              className={cn(
                "rounded-md px-2.5 py-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground",
                pathname.startsWith(n.href) && "bg-muted text-foreground",
              )}
            >
              {n.label}
            </Link>
          ))}
        </nav>
        <ThemeToggle />
        {!ready ? (
          // Auth state lives in localStorage; reserve space until it's read to avoid a flash.
          <div className="h-7 w-16" aria-hidden />
        ) : loggedIn ? (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              logout();
              router.push("/login");
            }}
          >
            <LogOut /> <span className="hidden sm:inline">Log out</span>
          </Button>
        ) : (
          <Button size="sm" onClick={() => router.push("/login")}>
            Log in
          </Button>
        )}
      </div>
    </header>
  );
}
