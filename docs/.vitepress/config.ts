import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(
  defineConfig({
    title: 'graphlet',
    description:
      'Graphlet analysis for petgraph: subgraph census, GDV/GDD, and network-motif detection',

    base: '/graphlet/',

    themeConfig: {
      nav: [
        { text: 'Guide', link: '/introduction' },
        { text: 'rhi', link: 'https://rhi.zone/' },
      ],

      sidebar: [
        {
          text: 'Guide',
          items: [
            { text: 'Introduction', link: '/introduction' },
            { text: 'Census substrate', link: '/census' },
            { text: 'Orbits (GDV/GDD)', link: '/orbits' },
            { text: 'Named motifs', link: '/motifs' },
            { text: 'Template matching', link: '/templates' },
          ],
        },
      ],

      socialLinks: [
        { icon: 'github', link: 'https://github.com/rhi-zone/graphlet' },
      ],

      search: {
        provider: 'local',
      },

      editLink: {
        pattern: 'https://github.com/rhi-zone/graphlet/edit/master/docs/:path',
        text: 'Edit this page on GitHub',
      },
    },

    vite: {
      optimizeDeps: {
        include: ['mermaid'],
      },
    },
  }),
)
