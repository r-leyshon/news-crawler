import { NextAuthOptions } from "next-auth"
import GitHubProvider from "next-auth/providers/github"

// Allowed GitHub username - only this user can delete articles
const ALLOWED_GITHUB_USERNAME = process.env.ALLOWED_GITHUB_USERNAME || "r-leyshon"

export const authOptions: NextAuthOptions = {
  providers: [
    GitHubProvider({
      clientId: process.env.GITHUB_ID ?? "",
      clientSecret: process.env.GITHUB_SECRET ?? "",
    }),
  ],
  callbacks: {
    async signIn({ profile }) {
      // Allow sign in for any GitHub user (read access)
      return true
    },
    async session({ session, token }) {
      // Add GitHub username to session
      if (session.user && token.sub) {
        session.user.id = token.sub
        // Check if user is the allowed owner
        const username = token.name as string | undefined
        session.user.isOwner = username?.toLowerCase() === ALLOWED_GITHUB_USERNAME.toLowerCase()
      }
      return session
    },
    async jwt({ token, profile }) {
      // Store GitHub username in token
      if (profile) {
        token.name = (profile as { login?: string }).login
      }
      return token
    },
  },
  pages: {
    signIn: "/api/auth/signin",
  },
}

// Extend the Session type to include our custom fields
declare module "next-auth" {
  interface Session {
    user: {
      id?: string
      name?: string | null
      email?: string | null
      image?: string | null
      isOwner?: boolean
    }
  }
}
